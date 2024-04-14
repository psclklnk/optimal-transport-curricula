import os
import pickle
import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Any, List, NoReturn, Callable


class TruncatedGaussianSampling:

    def __init__(self, delta: float, eta: float, context_bounds: Tuple[np.ndarray, np.ndarray]):
        context_exts = context_bounds[1] - context_bounds[0]
        self.delta = delta
        self.min_ret = None
        self.delta_stds = context_exts / 4
        self.min_stds = 0.005 * eta * np.ones(len(context_bounds[0]))
        self.context_bounds = context_bounds

    def get_data(self):
        return self.min_ret

    def set_data(self, data):
        self.min_ret = data

    def __call__(self, contexts, returns):
        if self.min_ret is None:
            self.min_ret = np.min(returns)

        self.min_ret = min(self.min_ret, np.min(returns))
        var_scales = np.clip(self.delta - returns, 0., np.inf) / (self.delta - self.min_ret)
        stds = self.min_stds[None, :] + var_scales[:, None] * self.delta_stds[None, :]

        contexts = contexts + np.random.normal(0, stds, size=contexts.shape)
        invalid = np.any(np.logical_or(contexts < self.context_bounds[0][None, :],
                                       contexts > self.context_bounds[1][None, :]), axis=-1)
        count = 0
        while np.any(invalid) and count < 10:
            new_noise = np.random.normal(0, stds[invalid], size=(np.sum(invalid), contexts.shape[1]))
            contexts[invalid, :] = contexts[invalid, :] + new_noise
            invalid = np.any(np.logical_or(contexts < self.context_bounds[0][None, :],
                                           contexts > self.context_bounds[1][None, :]), axis=-1)
            count += 1

        return np.clip(contexts, self.context_bounds[0], self.context_bounds[1])


class DiscreteSampling:

    def __init__(self, delta: float, max_explore_hops: int, neighbour_oracle: Callable[[np.ndarray], np.ndarray]):
        self.delta = delta
        self.min_ret = None
        self.max_explore_hops = max_explore_hops
        self.neighbour_oracle = neighbour_oracle

    def get_data(self):
        return self.min_ret

    def set_data(self, data):
        self.min_ret = data

    def __call__(self, contexts, returns):
        if self.min_ret is None:
            self.min_ret = np.min(returns)

        self.min_ret = min(self.min_ret, np.min(returns))
        var_scales = np.clip(self.delta - returns, 0., np.inf) / (self.delta - self.min_ret)
        explore_hops = var_scales * self.max_explore_hops
        explore_hops = np.where(np.random.rand(explore_hops.shape[0]) > 0.5, np.ceil(explore_hops),
                                np.floor(explore_hops)).astype(np.int64)

        done = False
        while not done:
            # Find out which for which context we still need to sample neighbours
            to_explore = np.where(explore_hops > 0)[0]
            for idx in to_explore:
                neighbours = self.neighbour_oracle(contexts[idx])
                sel_idx = np.random.randint(neighbours.shape[0])
                contexts[idx] = neighbours[sel_idx]
                explore_hops[idx] -= 1

            done = len(to_explore) == 0

        return contexts


class AbstractSuccessBuffer(ABC):

    def __init__(self, n: int, delta: float, sampler):
        self.max_size = n
        self.delta = delta
        self.contexts = None
        self.returns = np.array([-np.inf])
        self.sampler = sampler
        self.delta_reached = False

    @abstractmethod
    def update_delta_not_reached(self, new_contexts: np.ndarray, new_returns: np.ndarray,
                                 current_samples: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, List[bool]]:
        pass

    @abstractmethod
    def update_delta_reached(self, new_contexts: np.ndarray, new_returns: np.ndarray, current_samples: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, List[bool]]:
        pass

    def update(self, contexts, returns, current_samples):
        assert contexts.shape[0] < self.max_size

        if self.contexts is None:
            self.contexts = np.zeros((1, contexts.shape[1]), dtype=contexts.dtype)

        if not self.delta_reached:
            self.delta_reached, self.contexts, self.returns, mask = self.update_delta_not_reached(contexts, returns,
                                                                                                  current_samples)
        else:
            self.contexts, self.returns, mask = self.update_delta_reached(contexts, returns, current_samples)

        return contexts[mask, :], returns[mask]

    def read_train(self):
        return self.contexts.copy(), self.returns.copy()

    def read_update(self):
        # If we did not yet reach the desired threshold we enforce exploration by scaling the exploration noise w.r.t.
        # the distance to the desired threshold value
        if not self.delta_reached:
            offset = self.returns.shape[0] // 2
            sub_returns = self.returns[offset:]
            sub_contexts = self.contexts[offset:, :]

            # Do a resampling based on the achieved rewards (favouring higher rewards to be resampled)
            probs = sub_returns - self.returns[offset - 1]
            norm = np.sum(probs)
            if norm == 0:
                probs = np.ones(sub_returns.shape[0]) / sub_returns.shape[0]
            else:
                probs = probs / norm

            sample_idxs = np.random.choice(sub_returns.shape[0], self.max_size, p=probs)
            sampled_contexts = sub_contexts[sample_idxs, :]
            sampled_returns = sub_returns[sample_idxs]
        else:
            to_fill = self.max_size - self.returns.shape[0]
            add_idxs = np.random.choice(self.returns.shape[0], to_fill)
            sampled_contexts = np.concatenate((self.contexts, self.contexts[add_idxs, :]), axis=0)
            sampled_returns = np.concatenate((self.returns, self.returns[add_idxs]), axis=0)

        return self.sampler(sampled_contexts, sampled_returns)

    def get_data(self) -> Any:
        return None

    def set_data(self, data: Any) -> NoReturn:
        pass

    def save(self, path):
        with open(os.path.join(path, "teacher_success_buffer.pkl"), "wb") as f:
            pickle.dump((self.max_size, self.contexts, self.returns, self.delta_reached, self.get_data(),
                         self.sampler.get_data()), f)

    def load(self, path):
        with open(os.path.join(path, "teacher_success_buffer.pkl"), "rb") as f:
            self.max_size, self.contexts, self.returns, self.delta_reached, subclass_data, sampler_data = pickle.load(f)
        self.sampler.set_data(sampler_data)
        self.set_data(subclass_data)


class WassersteinSuccessBuffer(AbstractSuccessBuffer):

    def __init__(self, n: int, delta: float, sampler, squared_dist_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        super().__init__(n, delta, sampler)
        self.squared_dist_fn = squared_dist_fn

    def update_delta_not_reached(self, contexts: np.ndarray, returns: np.ndarray,
                                 current_samples: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, List[bool]]:
        # Only add samples that have a higher return than the median return in the buffer (we do >= here to allow
        # for binary rewards to work)
        med_idx = self.returns.shape[0] // 2
        mask = returns >= self.returns[med_idx]
        n_new = np.sum(mask)
        print("Improving buffer quality with %d samples" % n_new)

        # We do not want to shrink the buffer
        offset_idx = med_idx + 1
        if n_new < offset_idx:
            offset_idx = n_new

        new_returns = np.concatenate((returns[mask], self.returns[offset_idx:]), axis=0)
        new_contexts = np.concatenate((contexts[mask, :], self.contexts[offset_idx:, :]), axis=0)
        sort_idxs = np.argsort(new_returns)

        # Ensure that the buffer is only growing, never shrinking and that all the buffer sizes are consistent
        assert self.contexts.shape[0] <= new_contexts.shape[0]
        assert new_contexts.shape[0] == new_returns.shape[0]

        # These are the indices of the tasks that have NOT been added to the buffer (so the negation of the mas)
        rem_mask = ~mask

        # Ensure that we are not larger than the maximum size
        if new_returns.shape[0] > self.max_size:
            sort_idxs = sort_idxs[-self.max_size:]
            # Since we are clipping potentially removing some of the data chunks we need to update the remainder mask
            # Since we add the new samples at the beginning of the new buffers, we are interested whether the idxs
            # in [0, n_new) are still in the sort_idxs array. If this is NOT the case, then the sample has NOT been
            # added to the buffer.
            rem_mask[mask] = [i not in sort_idxs for i in np.arange(n_new)]

        new_delta_reached = self.returns[self.returns.shape[0] // 2] > self.delta
        return new_delta_reached, new_contexts[sort_idxs, :], new_returns[sort_idxs], rem_mask

    def update_delta_reached(self, contexts: np.ndarray, returns: np.ndarray, current_samples: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, List[bool]]:
        # Compute the new successful samples
        mask = returns >= self.delta
        n_new = np.sum(mask)

        if n_new > 0:
            remove_mask = self.returns < self.delta
            if not np.any(remove_mask) and self.returns.shape[0] >= self.max_size:
                extended_contexts = np.concatenate((self.contexts, contexts[mask, :]), axis=0)
                extended_returns = np.concatenate((self.returns, returns[mask]), axis=0)

                # At this stage we use the optimizer
                squared_dists = self.squared_dist_fn(extended_contexts[:, None, :], current_samples[None, :, :])
                assignments = linear_sum_assignment(squared_dists, maximize=False)

                ret_idxs = assignments[0]
                new_contexts = extended_contexts[ret_idxs, :]
                new_returns = extended_returns[ret_idxs]

                # We update the mask to indicate only the kept samples
                mask[mask] = [idx in (ret_idxs - self.contexts.shape[0]) for idx in np.arange(n_new)]

                print(f"Updated success buffer with {n_new} samples.")
            else:
                # We replace the unsuccessful samples by the successful ones
                if n_new < np.sum(remove_mask):
                    remove_idxs = np.argpartition(self.returns, kth=n_new)[:n_new]
                    remove_mask = np.zeros(self.returns.shape[0], dtype=bool)
                    remove_mask[remove_idxs] = True

                new_returns = np.concatenate((returns[mask], self.returns[~remove_mask]), axis=0)
                new_contexts = np.concatenate((contexts[mask, :], self.contexts[~remove_mask, :]), axis=0)

                if new_returns.shape[0] > self.max_size:
                    new_returns = new_returns[:self.max_size]
                    new_contexts = new_contexts[:self.max_size, :]

                # Ensure that the buffer is only growing, never shrinking and that all the buffer sizes are consistent
                assert self.contexts.shape[0] <= new_contexts.shape[0]
                assert new_contexts.shape[0] == new_returns.shape[0]
                print(f"Added {n_new} success samples to the success buffer.")
        else:
            new_contexts = self.contexts
            new_returns = self.returns

        return new_contexts, new_returns, ~mask


class AbstractSampler(ABC):

    def __init__(self, context_bounds: Tuple[np.ndarray, np.ndarray]):
        self.noise = 1e-3 * (context_bounds[1] - context_bounds[0])

    def update(self, context: np.ndarray, ret: float) -> NoReturn:
        pass

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return self.select(samples) + np.random.uniform(-self.noise, self.noise)

    @abstractmethod
    def select(self, samples: np.ndarray) -> np.ndarray:
        pass

    def save(self, path: str) -> NoReturn:
        pass

    def load(self, path: str) -> NoReturn:
        pass


class UniformSampler(AbstractSampler):

    def __init__(self, context_bounds: Tuple[np.ndarray, np.ndarray]):
        super(UniformSampler, self).__init__(context_bounds)

    def select(self, samples: np.ndarray) -> np.ndarray:
        return samples[np.random.randint(0, samples.shape[0]), :]


class DiscreteUniformSampler:

    def __init__(self):
        pass

    def update(self, context: np.ndarray, ret: float) -> NoReturn:
        pass

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return samples[np.random.randint(0, samples.shape[0]), :]

    def save(self, path: str) -> NoReturn:
        pass

    def load(self, path: str) -> NoReturn:
        pass
