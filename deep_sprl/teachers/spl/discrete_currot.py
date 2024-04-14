import os
import time
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Callable
from scipy.optimize import linear_sum_assignment
from deep_sprl.teachers.util import NadarayaWatsonPy
from deep_sprl.teachers.abstract_teacher import AbstractTeacher
from deep_sprl.teachers.spl.wasserstein_interpolation import SamplingDiscreteWassersteinInterpolation
from deep_sprl.teachers.spl.currot_utils import WassersteinSuccessBuffer, DiscreteSampling, DiscreteUniformSampler


class DiscreteGradient(AbstractTeacher):

    def __init__(self, init_samples: np.ndarray, target_sampler: Callable[[int], np.ndarray], perf_lb: float,
                 epsilon: float, distance_function: Callable, candidates: np.ndarray):
        self.alpha = 0.
        self.perf_lb = perf_lb
        self.epsilon = epsilon
        self.current_distribution = init_samples
        self.distance_function = distance_function
        self.target_sampler = target_sampler
        self.candidates = candidates

        self.distribution_trace = [np.copy(self.current_distribution)]
        self.alpha_trace = [0.]

    def update_distribution(self, contexts: np.ndarray, returns: np.ndarray):
        if np.mean(returns) >= self.perf_lb:
            # Update the samples
            n_samples = self.current_distribution.shape[0]
            target_samples = self.target_sampler(n_samples)

            if self.alpha < 1:
                self.alpha = min(1., self.alpha + self.epsilon)

                dist_mat = self.distance_function(self.current_distribution[:, None], target_samples[None, :])
                rows, cols = linear_sum_assignment(dist_mat)

                source_dists = self.distance_function(self.current_distribution[rows[:, None]],
                                                      self.candidates[None, :]).astype(np.float32)
                target_dists = self.distance_function(target_samples[rows[:, None]],
                                                      self.candidates[None, :]).astype(np.float32)
                best_cand_idx = np.argmin(self.alpha * (target_dists ** 2) + (1 - self.alpha) * (source_dists ** 2),
                                          axis=-1)

                self.current_distribution = self.candidates[best_cand_idx]
                self.distribution_trace.append(np.copy(self.current_distribution))
                self.alpha_trace.append(self.alpha)
            else:
                self.current_distribution = target_samples

            print(f"Alpha: {self.alpha}")
        else:
            print(f"Agent not proficient enough: {np.mean(returns)} vs {self.perf_lb}")

    def sample(self):
        return self.current_distribution[np.random.randint(0, self.current_distribution.shape[0])]

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
            pickle.dump((self.alpha, self.current_distribution), f)

    def load(self, path):
        with open(os.path.join(path, "teacher.pkl"), "rb") as f:
            self.alpha, self.current_distribution = pickle.load(f)

        self.alpha_trace = []
        self.distribution_trace = []


class DiscreteCurrOT(AbstractTeacher):

    def __init__(self, init_samples: np.ndarray, target_sampler: Callable[[int], np.ndarray], perf_lb: float,
                 epsilon: float, neighbour_oracle: Callable[[np.ndarray], np.ndarray],
                 distance_function: Callable, callback=None, wait_until_threshold=False, max_sample_hops: int = 4):
        self.threshold_reached = False
        self.epsilon = epsilon
        self.wait_until_threshold = wait_until_threshold
        # init_samples, target_sampler, neighbour_oracle, distance_function, perf_lb, callback=None)
        self.teacher = SamplingDiscreteWassersteinInterpolation(init_samples, target_sampler, neighbour_oracle,
                                                                distance_function, perf_lb, callback=callback)
        self.squared_distance_fn = lambda x, y: distance_function(x, y).astype(float) ** 2
        self.success_buffer = WassersteinSuccessBuffer(init_samples.shape[0], perf_lb,
                                                       DiscreteSampling(perf_lb, max_sample_hops, neighbour_oracle),
                                                       self.squared_distance_fn)
        self.fail_context_buffer = []
        self.fail_return_buffer = []
        self.sampler = DiscreteUniformSampler()

    def on_rollout_end(self, context, ret):
        self.sampler.update(context, ret)

    def update_distribution(self, contexts, returns):
        t_up1 = time.time()
        fail_contexts, fail_returns = self.success_buffer.update(contexts, returns,
                                                                 self.teacher.target_sampler(
                                                                     self.teacher.current_samples.shape[0]))

        if self.threshold_reached:
            self.fail_context_buffer.extend(fail_contexts)
            self.fail_context_buffer = self.fail_context_buffer[-self.teacher.n_samples:]
            self.fail_return_buffer.extend(fail_returns)
            self.fail_return_buffer = self.fail_return_buffer[-self.teacher.n_samples:]

        success_contexts, success_returns = self.success_buffer.read_train()
        if len(self.fail_context_buffer) == 0:
            train_contexts = success_contexts
            train_returns = success_returns
        else:
            train_contexts = np.concatenate((np.stack(self.fail_context_buffer, axis=0), success_contexts), axis=0)
            train_returns = np.concatenate((np.stack(self.fail_return_buffer, axis=0), success_returns), axis=0)

        model = NadarayaWatsonPy(train_contexts, train_returns, 0.3 * self.epsilon,
                                 custom_distance_function=self.squared_distance_fn)
        t_up2 = time.time()

        t_mo1 = time.time()
        avg_perf = np.mean(model.predict_individual(self.teacher.current_samples))
        if self.threshold_reached or avg_perf >= self.teacher.perf_lb:
            self.threshold_reached = True
            self.teacher.update_distribution(model, self.success_buffer.read_update())
        else:
            print(f"Current performance: {avg_perf} vs {self.teacher.perf_lb}")
            if self.wait_until_threshold:
                print("Not updating sampling distribution, as performance threshold not met")
            else:
                self.teacher.current_samples = self.success_buffer.read_update()
        t_mo2 = time.time()

        print(f"Total update took: {t_mo2 - t_up1} (Buffer/Update: {t_up2 - t_up1}/{t_mo2 - t_mo1})")

    def sample(self):
        return self.sampler(self.teacher.current_samples)

    def save(self, path):
        self.teacher.save(path)
        self.success_buffer.save(path)
        self.sampler.save(path)

    def load(self, path):
        self.teacher.load(path)
        self.success_buffer.load(path)
        self.sampler.load(path)
