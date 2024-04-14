import os
import torch
import pickle
import numpy as np
from geomloss import SamplesLoss
from scipy.optimize import linear_sum_assignment


class SamplingWassersteinInterpolation:

    def __init__(self, init_samples, target_sampler, perf_lb, epsilon, bounds, callback=None):
        self.current_samples = init_samples
        self.n_samples, self.dim = self.current_samples.shape
        self.target_sampler = target_sampler
        self.bounds = bounds
        self.perf_lb = perf_lb
        self.epsilon = epsilon
        self.callback = callback

    def sample_ball(self, targets, samples=None, half_ball=None, n=100, add_line_samples: bool = True):
        if samples is None:
            samples = self.current_samples

        # Taken from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        # Method 20
        direction = np.random.normal(0, 1, (n, self.dim))
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        r = np.power(np.random.uniform(size=(n, 1)), 1. / self.dim)

        # We only consider samples that decrease the distance objective (i.e. are aligned with the direction)
        noise = r * (direction / norm)
        dirs = targets - samples
        dir_norms = np.einsum("ij,ij->i", dirs, dirs)
        noise_projections = np.einsum("ij,kj->ik", dirs / dir_norms[:, None], noise)

        projected_noise = np.where((noise_projections > 0)[..., None], noise[None, ...],
                                   noise[None, ...] - 2 * noise_projections[..., None] * dirs[:, None, :])
        if half_ball is not None:
            projected_noise[~half_ball] = noise

        scales = np.minimum(self.epsilon, np.sqrt(dir_norms))[:, None, None]

        if add_line_samples:
            normed_dirs = (dirs / np.sqrt(dir_norms[:, None]))
            ext_samples = samples[:, None, :] + \
                          np.linspace(0, scales[:, :, 0], 11, axis=1)[:, 1:, :] * normed_dirs[:, None, :]
            return np.clip(np.concatenate((samples[..., None, :] + scales * projected_noise, ext_samples), axis=1),
                           self.bounds[0], self.bounds[1])
        else:
            return np.clip(samples[..., None, :] + scales * projected_noise, self.bounds[0], self.bounds[1])

    @staticmethod
    def visualize_particles(init_samples, particles, performances):
        if particles.shape[-1] != 2:
            raise RuntimeError("Can only visualize 2D data")

        import matplotlib.pyplot as plt
        f = plt.figure()
        ax = f.gca()
        scat = ax.scatter(particles[0, :, 0], particles[0, :, 1], c=performances[0, :])
        ax.scatter(init_samples[0, 0], init_samples[0, 1], marker="x", c="red")
        plt.colorbar(scat)
        plt.show()

    def ensure_successful_initial(self, model, init_samples, success_samples):
        squared_dists = np.sum(np.square(init_samples[:, None, :] - success_samples[None, :, :]), axis=-1)
        success_assignment = linear_sum_assignment(squared_dists, maximize=False)

        performance_reached = model.predict_individual(init_samples) >= self.perf_lb
        assigned_samples = success_samples[success_assignment[1][~performance_reached]]
        init_samples[~performance_reached, :] = assigned_samples
        performance_reached[~performance_reached] = model.predict_individual(assigned_samples) >= self.perf_lb

        return init_samples, performance_reached

    def update_distribution(self, model, success_samples, debug=False):
        init_samples, performance_reached = self.ensure_successful_initial(model, self.current_samples.copy(),
                                                                           success_samples)
        target_samples = self.target_sampler(self.n_samples)
        if debug:
            target_samples_true = target_samples.copy()
        assignments = linear_sum_assignment(np.sum(np.square(init_samples[:, None] - target_samples[None, :]), axis=-1))
        init_samples = init_samples[assignments[0]]
        target_samples = target_samples[assignments[1]]
        particles = self.sample_ball(target_samples, samples=init_samples, half_ball=performance_reached)

        distances = np.linalg.norm(particles - target_samples[:, None, :], axis=-1)
        performances = model.predict_individual(particles)
        if debug:
            self.visualize_particles(init_samples, particles, performances)

        mask = performances > self.perf_lb
        solution_possible = np.any(mask, axis=-1)
        distances[~mask] = np.inf
        opt_idxs = np.where(solution_possible, np.argmin(distances, axis=-1), np.argmax(performances, axis=-1))
        new_samples = particles[np.arange(0, self.n_samples), opt_idxs]

        print(f"New Wasserstein Distance: {np.sqrt(np.mean(np.sum(np.square(new_samples - target_samples), axis=-1)))}")

        if debug:
            vis_idxs = np.random.randint(0, target_samples.shape[0], size=50)
            import matplotlib.pyplot as plt
            xs, ys = np.meshgrid(np.linspace(0, 9, num=150), np.linspace(0, 6, num=100))
            zs = model.predict_individual(np.stack((xs, ys), axis=-1))
            ims = plt.imshow(zs, extent=[0, 9, 0, 6], origin="lower")
            plt.contour(xs, ys, zs, [180])
            plt.colorbar(ims)

            plt.scatter(target_samples_true[vis_idxs, 0], target_samples_true[vis_idxs, 1], marker="x", color="red")
            plt.scatter(self.current_samples[vis_idxs, 0], self.current_samples[vis_idxs, 1], marker="o", color="C0")
            plt.scatter(init_samples[vis_idxs, 0], init_samples[vis_idxs, 1], marker="o", color="C2")
            plt.scatter(new_samples[vis_idxs, 0], new_samples[vis_idxs, 1], marker="o", color="C1")
            plt.xlim([0, 9])
            plt.ylim([0, 6])
            plt.show()

        if self.callback is not None:
            self.callback(self.current_samples, new_samples, success_samples, target_samples)

        self.current_samples = new_samples

    def save(self, path):
        with open(os.path.join(path, "teacher.pkl"), "wb") as f:
            pickle.dump((self.current_samples, self.perf_lb, self.epsilon), f)

    def load(self, path):
        with open(os.path.join(path, "teacher.pkl"), "rb") as f:
            tmp = pickle.load(f)

            self.current_samples = tmp[0]
            self.n_samples = self.current_samples.shape[0]

            self.perf_lb = tmp[1]
            self.epsilon = tmp[2]


class SamplingDiscreteWassersteinInterpolation:

    def __init__(self, init_samples, target_sampler, neighbour_oracle, distance_function, perf_lb, callback=None):
        self.current_samples = init_samples
        self.neighbour_orcale = neighbour_oracle
        self.n_samples = self.current_samples.shape[0]
        self.target_sampler = target_sampler

        self.distance_function = distance_function
        self.perf_lb = perf_lb
        self.callback = callback

    def ensure_successful_initial(self, model, success_samples):
        dists = self.distance_function(self.current_samples[:, None, :], success_samples[None, :])
        # Make sure that we do not overflow if we square the values
        success_assignment = linear_sum_assignment(dists.astype(float) ** 2, maximize=False)

        valid_assignments = dists[success_assignment[0], success_assignment[1]] < np.iinfo(dists.dtype).max
        performance_reached = model.predict_individual(self.current_samples) >= self.perf_lb
        assigned_samples = success_samples[success_assignment[1][~performance_reached], :]
        self.current_samples[~performance_reached] = np.where(valid_assignments[~performance_reached, None],
                                                              assigned_samples,
                                                              self.current_samples[~performance_reached])

    @staticmethod
    def argmin(distances):
        candidates = np.where(np.isclose(np.min(distances), distances))[0]
        return candidates[np.random.randint(candidates.shape[0])]

    @staticmethod
    def argmax(distances):
        candidates = np.where(np.isclose(np.max(distances), distances))[0]
        return candidates[np.random.randint(candidates.shape[0])]

    def update_distribution(self, model, success_samples):
        self.ensure_successful_initial(model, success_samples)
        target_samples = self.target_sampler(self.n_samples)

        # Reorder to target samples such that particles need to be minimally moved
        dists = self.distance_function(self.current_samples[:, None], target_samples[None, :])
        # Make sure that we do not overflow if we square the values
        __, target_idxs = linear_sum_assignment(dists.astype(float) ** 2, maximize=False)
        target_samples = target_samples[target_idxs]

        # We generate the new samples
        new_samples = []
        for current_sample, target_sample in zip(self.current_samples, target_samples):
            neighbours = self.neighbour_orcale(current_sample)
            target_distances = np.squeeze(self.distance_function(target_sample[None, None, :], neighbours[None, :, :]))
            performances = model.predict_individual(neighbours)
            solution_possible = np.any(performances > self.perf_lb)
            best_idxs = self.argmin(target_distances) if solution_possible else self.argmax(performances)
            new_samples.append(neighbours[best_idxs])

        if self.callback is not None:
            self.callback(self.current_samples, new_samples, success_samples, target_samples)

        self.current_samples = np.array(new_samples)
        new_dists = self.distance_function(self.current_samples[:, None, :], target_samples[:, None, :])[:, 0]
        assert not np.any(new_dists == np.iinfo(new_dists.dtype).max)
        print(f"New Distance: {np.sqrt(np.mean(new_dists.astype(float) ** 2))}")

    def save(self, path):
        with open(os.path.join(path, "teacher.pkl"), "wb") as f:
            pickle.dump((self.current_samples, self.perf_lb), f)

    def load(self, path):
        with open(os.path.join(path, "teacher.pkl"), "rb") as f:
            tmp = pickle.load(f)

            self.current_samples = tmp[0]
            self.n_samples = self.current_samples.shape[0]
            self.perf_lb = tmp[1]


class UnBiasedSinkhornBaryCenter:

    def __init__(self):
        pass

    def __call__(self, alpha: float, source: np.ndarray, target: np.ndarray, convergence_tol: float = 1e-3,
                 lr: float = 5e-1, blur: float = 1e-2):
        loss = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=.95)

        # Make sure that we won't modify the reference samples
        x_i, y_j = torch.from_numpy(source).clone(), torch.from_numpy(target).clone()
        x_interp = torch.from_numpy(source).clone()

        # We're going to perform gradient descent on Loss(α, β)
        # wrt. the positions x_i of the diracs masses that make up α:
        x_interp.requires_grad = True

        done = False
        while not done:
            # Compute cost and gradient
            l_αβ = (1 - alpha) * loss(x_i, x_interp) + alpha * loss(y_j, x_interp)
            [g] = torch.autograd.grad(l_αβ, [x_interp])

            # in-place modification of the tensor's values
            x_interp_new = x_interp - lr * x_interp.shape[0] * g
            done = torch.mean(torch.abs(x_interp_new - x_interp)) < convergence_tol

            x_interp.data -= lr * len(x_interp) * g

        return x_interp.detach().numpy()
