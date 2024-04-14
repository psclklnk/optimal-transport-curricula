import os
import scipy
import pickle
import cvxpy as cp
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Callable
from scipy.optimize import linprog
from deep_sprl.teachers.abstract_teacher import AbstractTeacher


def generate_wasserstein_constraint(n, m):
    a1 = np.zeros((n, n * m))
    a2 = np.zeros((m, n * m))
    for i in range(n):
        for j in range(m):
            a1[i, i * m + j] = 1
            a2[j, i * m + j] = 1

    return np.concatenate((a1, a2), axis=0)


def generate_wasserstein_barycenter_constraint(n_p1, n_p2, n_q):
    # We have three optimization variables: (n_q x n_p1), (n_q x n_p2), n_q
    n_var = n_q * (n_p1 + n_p2 + 1)

    # We generate the marginal constraints for p1, p2 and q
    a = scipy.linalg.block_diag(generate_wasserstein_constraint(n_p1, n_q), generate_wasserstein_constraint(n_p2, n_q))

    # For the constraints that relate to q, we need to subtract q by adding n_q new rows to a
    a = np.concatenate((a, np.zeros((a.shape[0], n_q))), axis=-1)
    assert a.shape[1] == n_var
    a[n_p1:n_p1 + n_q, -n_q:] = -np.eye(n_q)
    a[n_p1 + n_q + n_p2:n_p1 + n_q + n_p2 + n_q, -n_q:] = -np.eye(n_q)

    return a


class PerformanceNotReachedError(RuntimeError):

    def __init__(self, text):
        super().__init__(text)


class WassersteinDistance:

    def __init__(self, n, m=None):
        if m is None:
            m = n

        self.a = generate_wasserstein_constraint(n, m)
        self.bounds = [(0, None)] * (n * m)

    def __call__(self, metric, d1, d2):
        c = np.reshape(metric, -1)
        if np.any(np.isclose(d1, 1.)) or np.any(np.isclose(d2, 1.)):
            return np.sum(metric * (d1[:, None] * d2[None, :]))
        else:
            res = linprog(c, A_eq=self.a, b_eq=np.concatenate((d1, d2), axis=0), bounds=self.bounds, method="highs-ds")

            if res.fun is None:
                raise RuntimeError("Error! Success: %s, Error Code: %d" % (str(res.success), res.status))

            return res.fun


class ConstrainedEntropicWassersteinBarycenter:

    def __init__(self, delta_perf, ent_lb, mu):
        self.delta_perf = delta_perf
        self.ent_lb = ent_lb
        self.mu = mu
        self.mu_mask = mu > 0
        self.n_mu = np.sum(self.mu_mask)

    def __call__(self, metric, performances):
        # We first "remove" the contexts in which the performance constraint is not met
        perf_reached = performances >= self.delta_perf
        if not np.any(perf_reached):
            raise PerformanceNotReachedError("Performance level is not reached anywhere")

        n_p = np.sum(perf_reached)

        # We create the marginalization constraints fur mu and the new context distribution
        a = generate_wasserstein_constraint(self.n_mu, n_p)
        # Since we optimize over n_p, we need to add n_p additional variables to the n_mu x n_np variables
        a = np.concatenate((a, np.zeros((a.shape[0], n_p))), axis=1)
        # Since we optimize over n_p, the marginalization constraint w.r.t n_p should be zero if we subtract n_p
        a[self.n_mu:, -n_p:] = -np.eye(n_p)

        con_val = np.concatenate((self.mu[self.mu_mask], np.zeros(n_p)))

        c = np.zeros(a.shape[1])
        c[:self.n_mu * n_p] = np.reshape(metric[self.mu_mask, :][:, perf_reached], -1)
        # We standardize the range of the values such that they are in [0, 10]. This normalization helped with
        # numerical issues of the underlying solver
        scale = 10 / np.max(c)
        c *= scale

        x = cp.Variable(a.shape[1])
        prob = cp.Problem(cp.Minimize(c.T @ x), [0 <= x, a @ x == con_val, self.ent_lb <= cp.sum(cp.entr(x[-n_p:]))])
        try:
            prob.solve(verbose=False)
        except cp.error.SolverError:
            print("ECOS solver failed, running with SCS")
            prob.solve(solver=cp.SCS, eps=1e-3)

        if prob.status == "infeasible":
            raise PerformanceNotReachedError("No solution to the optimization problem found")

        p = np.zeros_like(self.mu)
        p[perf_reached] = np.clip(x.value[-n_p:], 0, np.inf)

        # We re-normalize the result
        return prob.value / scale, p / np.sum(p)


class ConstrainedWassersteinBarycenter:

    def __init__(self, delta, mu):
        self.delta = delta
        self.mu = mu
        self.mu_mask = mu > 0
        self.n_mu = np.sum(self.mu_mask)

    def __call__(self, metric, performances):
        # We first "remove" the contexts in which the performance constraint is not met
        perf_reached = performances >= self.delta
        if not np.any(perf_reached):
            raise PerformanceNotReachedError("Performance level is not reached anywhere")

        n_p = np.sum(perf_reached)

        # We create the marginalization constraints fur mu and the new context distribution
        a = generate_wasserstein_constraint(self.n_mu, n_p)
        # Since we optimize over n_p, we need to add n_p additional variables to the n_mu x n_np variables
        a = np.concatenate((a, np.zeros((a.shape[0], n_p))), axis=1)
        # Since we optimize over n_p, the marginalization constraint w.r.t n_p should be zero if we subtract n_p
        a[self.n_mu:, -n_p:] = -np.eye(n_p)

        con_val = np.concatenate((self.mu[self.mu_mask], np.zeros(n_p)))

        c = np.zeros(a.shape[1])
        c[:self.n_mu * n_p] = np.reshape(metric[self.mu_mask, :][:, perf_reached], -1)

        bounds = [(0, None) for _ in range(a.shape[1])]
        res = linprog(c, A_eq=a, b_eq=con_val, bounds=bounds, method="highs-ds")

        if res.fun is None:
            raise PerformanceNotReachedError("No solution to the optimization problem found")

        p = np.zeros_like(self.mu)
        p[perf_reached] = res.x[-n_p:]

        return res.fun, p


class EntropicWassersteinBarycenter:

    def __init__(self, ent_eps, mu):
        self.ent_eps = ent_eps
        self.mu = mu
        self.mu_mask = mu > 0
        self.n_mu = np.sum(self.mu_mask)

    def __call__(self, metric: np.ndarray, alpha: float, q: np.ndarray):
        n_p = self.mu.shape[0]
        q_mask = q > 0
        n_q = np.sum(q_mask)

        # We create the marginalization constraints
        a = generate_wasserstein_barycenter_constraint(self.n_mu, n_q, n_p)
        con_val = np.concatenate((self.mu[self.mu_mask], np.zeros(n_p), q[q_mask], np.zeros(n_p)))

        # Next we generate the wasserstein distance constraint on q
        c = np.zeros(a.shape[1])
        c[:self.n_mu * n_p] = alpha * np.reshape(metric[self.mu_mask, :], -1)
        c[self.n_mu * n_p:-n_p] = (1 - alpha) * np.reshape(metric[q_mask, :], -1)

        # g = -np.eye(a.shape[1])
        # h = np.zeros(a.shape[1])

        x = cp.Variable(a.shape[1])
        ent_scale = self.ent_eps / max(np.max(metric[self.mu_mask, :]), np.max(metric[q_mask, :]))
        prob = cp.Problem(cp.Minimize(c.T @ x - ent_scale * cp.sum(cp.entr(x[:self.n_mu * n_p])) -
                                      ent_scale * cp.sum(cp.entr(x[self.n_mu * n_p:-n_p]))),
                          [0 <= x, a @ x == con_val])
        try:
            prob.solve(verbose=False)
        except cp.error.SolverError:
            print("ECOS solver failed, running with SCS")
            prob.solve(solver=cp.SCS, eps=1e-3)

        if prob.status == "infeasible":
            raise PerformanceNotReachedError("No solution to the optimization problem found")

        p = np.clip(x.value[-n_p:], 0, np.inf)
        return p / np.sum(p)


class WassersteinBarycenter:

    def __init__(self, mu):
        self.mu = mu
        self.mu_mask = mu > 0
        self.n_mu = np.sum(self.mu_mask)

    def __call__(self, metric: np.ndarray, alpha: float, q: np.ndarray):
        n_p = self.mu.shape[0]
        q_mask = q > 0

        # We create the marginalization constraints
        a = generate_wasserstein_barycenter_constraint(self.n_mu, np.sum(q_mask), n_p)
        con_val = np.concatenate((self.mu[self.mu_mask], np.zeros(n_p), q[q_mask], np.zeros(n_p)))

        # Next we generate the wasserstein distance constraint on q
        c = np.zeros(a.shape[1])
        # (1 - alpha) times Wasserstein distance to q
        c[self.n_mu * n_p:-n_p] = (1 - alpha) * np.reshape(metric[q_mask, :], -1)
        # alpha times Wasserstein distance to mu
        c[:self.n_mu * n_p] = alpha * np.reshape(metric[self.mu_mask, :], -1)

        bounds = [(0, None) for _ in range(a.shape[1])]
        res = linprog(c, A_eq=a, b_eq=con_val, bounds=bounds, method="highs-ds")

        if res.fun is None:
            raise PerformanceNotReachedError("No solution to the optimization problem found")

        return res.x[-n_p:]


class ExactCurrOT(AbstractTeacher):

    def __init__(self, init_dist: np.ndarray, dist_fn: Callable, performance_model, delta: float,
                 mu: np.ndarray, ent_lb: float = 0.):
        self.dist_fn = dist_fn
        self.performance_model = performance_model
        self.current_distribution = init_dist
        if ent_lb == 0:
            self.barycenter_computer = ConstrainedWassersteinBarycenter(delta, mu)
        else:
            self.barycenter_computer = ConstrainedEntropicWassersteinBarycenter(delta, ent_lb, mu)
        self.distribution_trace = [np.copy(init_dist)]

    def update_distribution(self, contexts: np.ndarray, returns: np.ndarray):
        self.performance_model.update(contexts, returns)
        expected_returns = self.performance_model()
        distance = self.dist_fn()

        try:
            dist, self.current_distribution = self.barycenter_computer(distance ** 2, expected_returns)
            print(f"New Distance: {dist}")
        except PerformanceNotReachedError:
            print("Could not update distribution. Computing close sufficient distribution")
            eligible_idxs = np.where(expected_returns >= self.barycenter_computer.delta)[0]
            replacements = eligible_idxs[
                np.argmin(distance[self.current_distribution > 0, :][:, eligible_idxs], axis=-1)]
            p_new = np.zeros_like(self.current_distribution)
            np.add.at(p_new, replacements, self.current_distribution[self.current_distribution > 0])
            self.current_distribution = p_new

        self.distribution_trace.append(np.copy(self.current_distribution))

    def sample(self):
        return np.argmax(np.random.uniform(0., 1.) <= np.cumsum(self.current_distribution))

    def save(self, path):
        base_path = Path(path).parent
        trace_path = (base_path / "context_trace.pkl")

        if trace_path.exists():
            with open(trace_path, "rb") as f:
                old_pt = pickle.load(f)

            with open(trace_path, "wb") as f:
                pickle.dump(old_pt + self.distribution_trace, f)
        else:
            with open(trace_path, "wb") as f:
                pickle.dump(self.distribution_trace, f)
        self.distribution_trace = []

        with open(os.path.join(path, "teacher.pkl"), "wb") as f:
            pickle.dump(self.current_distribution, f)

    def load(self, path):
        with open(os.path.join(path, "teacher.pkl"), "rb") as f:
            self.current_distribution = pickle.load(f)

        self.distribution_trace = []


class ExactGradient(AbstractTeacher):

    def __init__(self, init_dist: np.ndarray, dist_fn: Callable, performance_model, delta: float,
                 mu: np.ndarray, ent_eps: float = 0):
        self.delta = delta
        self.performance_model = performance_model
        self.dist_fn = dist_fn
        self.init_dist = init_dist
        self.current_distribution = init_dist

        self.distribution_trace = [np.copy(self.init_dist)]
        self.alpha_trace = [0.]
        if ent_eps == 0:
            self.barycenter_computer = WassersteinBarycenter(mu)
        else:
            self.barycenter_computer = EntropicWassersteinBarycenter(ent_eps, mu)

        self.prec_alphas = np.linspace(0, 1, 25)
        self.last_distance = self.dist_fn()
        if self.last_distance is not None:
            print("Pre-Computing Barycenters")
            self.interpolations = np.array([self.barycenter_computer(self.last_distance ** 2, alpha, init_dist)
                                            for alpha in tqdm(self.prec_alphas)])

    def update_distribution(self, contexts: np.ndarray, returns: np.ndarray):
        self.performance_model.update(contexts, returns)
        expected_returns = self.performance_model()

        distance = self.dist_fn()
        if self.last_distance is None or not np.all(distance == self.last_distance):
            self.last_distance = distance
            print("Computing Barycenters")
            self.interpolations = np.array([self.barycenter_computer(distance ** 2, alpha, self.init_dist) for alpha in
                                            tqdm(self.prec_alphas)])

        dist_returns = np.sum(self.interpolations * expected_returns, axis=-1)
        if not np.any(dist_returns >= self.delta):
            opt_alpha = 0.
            self.current_distribution = self.init_dist
        else:
            valid_idxs = np.where(dist_returns >= self.delta)[0]
            max_idx = np.max(valid_idxs)
            if self.prec_alphas[max_idx] == 1.:
                self.current_distribution = self.interpolations[max_idx]
                opt_alpha = 1.
            else:
                from scipy.optimize import brentq
                opt_alpha = brentq(
                    lambda alpha: np.sum(expected_returns * self.barycenter_computer(distance ** 2, alpha,
                                                                                     self.init_dist)) - self.delta,
                    self.prec_alphas[max_idx], self.prec_alphas[max_idx + 1], xtol=1e-3, rtol=1e-3)
                self.current_distribution = self.barycenter_computer(distance ** 2, opt_alpha, self.init_dist)

        self.distribution_trace.append(np.copy(self.current_distribution))
        self.alpha_trace.append(opt_alpha)

        print(f"Alpha: {opt_alpha}")

    def sample(self):
        return np.argmax(np.random.uniform(0., 1.) <= np.cumsum(self.current_distribution))

    def save(self, path):
        base_path = Path(path).parent
        trace_path = (base_path / "context_trace.pkl")

        if trace_path.exists():
            with open(trace_path, "rb") as f:
                old_at, old_pt = pickle.load(f)

            with open(trace_path, "wb") as f:
                pickle.dump((old_at + self.alpha_trace, old_pt + self.distribution_trace), f)
        else:
            with open(trace_path, "wb") as f:
                pickle.dump((self.alpha_trace, self.distribution_trace), f)
        self.alpha_trace = []
        self.distribution_trace = []

        with open(os.path.join(path, "teacher.pkl"), "wb") as f:
            pickle.dump(self.current_distribution, f)

    def load(self, path):
        with open(os.path.join(path, "teacher.pkl"), "rb") as f:
            self.current_distribution = pickle.load(f)

        self.alpha_trace = []
        self.distribution_trace = []
