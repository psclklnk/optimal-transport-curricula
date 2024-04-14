import os
import time
import pickle
import numpy as np
from pathlib import Path
from typing import Callable, Tuple
from deep_sprl.teachers.util import NadarayaWatson
from deep_sprl.teachers.abstract_teacher import AbstractTeacher
from deep_sprl.teachers.spl.wasserstein_interpolation import SamplingWassersteinInterpolation, \
    UnBiasedSinkhornBaryCenter
from deep_sprl.teachers.spl.currot_utils import WassersteinSuccessBuffer, TruncatedGaussianSampling, UniformSampler


class CurrOT(AbstractTeacher):

    def __init__(self, context_bounds, init_samples, target_sampler, perf_lb, epsilon, callback=None,
                 wait_until_threshold=False):
        self.context_bounds = context_bounds
        self.threshold_reached = False
        self.wait_until_threshold = wait_until_threshold
        self.teacher = SamplingWassersteinInterpolation(init_samples, target_sampler, perf_lb, epsilon,
                                                        self.context_bounds, callback=callback)
        # n: int, delta: float, sampler, dist_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
        self.success_buffer = WassersteinSuccessBuffer(init_samples.shape[0], perf_lb,
                                                       TruncatedGaussianSampling(perf_lb, epsilon, context_bounds),
                                                       lambda x, y: np.sum(np.square(x - y), axis=-1))
        self.fail_context_buffer = []
        self.fail_return_buffer = []
        self.sampler = UniformSampler(self.context_bounds)

    def on_rollout_end(self, context, ret):
        self.sampler.update(context, ret)

    def update_distribution(self, contexts, returns):
        t_up1 = time.time()
        target_samples = self.teacher.target_sampler(self.teacher.current_samples.shape[0])
        fail_contexts, fail_returns = self.success_buffer.update(contexts, returns, target_samples)

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
        model = NadarayaWatson(train_contexts, train_returns, 0.3 * self.teacher.epsilon)
        t_up2 = time.time()

        t_mo1 = time.time()
        avg_perf = np.mean(model.predict_individual(self.teacher.current_samples))
        if self.threshold_reached or avg_perf >= self.teacher.perf_lb:
            self.threshold_reached = True
            self.teacher.update_distribution(model, self.success_buffer.read_update())
        else:
            print("Current performance: %.3e vs %.3e" % (avg_perf, self.teacher.perf_lb))
            if self.wait_until_threshold:
                print("Not updating sampling distribution, as performance threshold not met")
            else:
                self.teacher.current_samples = self.success_buffer.read_update()
        t_mo2 = time.time()

        print("Total update took: %.3e (Buffer/Update: %.3e/%.3e)" % (t_mo2 - t_up1, t_up2 - t_up1, t_mo2 - t_mo1))

    def sample(self):
        sample = self.sampler(self.teacher.current_samples)
        return np.clip(sample, self.context_bounds[0], self.context_bounds[1])

    def save(self, path):
        self.teacher.save(path)
        self.success_buffer.save(path)
        self.sampler.save(path)

    def load(self, path):
        self.teacher.load(path)
        self.success_buffer.load(path)
        self.sampler.load(path)


class Gradient(AbstractTeacher):

    def __init__(self, context_bounds: Tuple[np.ndarray, np.ndarray], init_samples: np.ndarray,
                 target_sampler: Callable[[int], np.ndarray], eps: float, delta: float, ent_eps: float = 1e-3,
                 optimize_initial_samples: bool = False):
        self.init_samples = init_samples
        self.current_distribution = np.copy(init_samples)
        self.target_sampler = target_sampler
        self.eps = eps
        self.ent_eps = ent_eps
        self.alpha = 0.
        self.delta = delta
        self.interpolator = UnBiasedSinkhornBaryCenter()

        self.alpha_trace = [0.]
        self.distribution_trace = [np.copy(init_samples)]
        if optimize_initial_samples:
            self.success_buffer = WassersteinSuccessBuffer(init_samples.shape[0], delta,
                                                           TruncatedGaussianSampling(delta, 0.01, context_bounds),
                                                           lambda x, y: np.sum(np.square(x[:, None] - y[None, :]),
                                                                               axis=-1))
            self.interpolation_started = False
        else:
            self.success_buffer = None
            self.interpolation_started = True

    def update_distribution(self, contexts: np.ndarray, returns: np.ndarray):
        print(f"Current performance: {np.mean(returns): .3e} vs {self.delta: .3e}")

        if self.interpolation_started:
            if np.mean(returns) >= self.delta:
                if self.alpha < 1:
                    self.alpha = min(1., self.alpha + self.eps)
                    self.current_distribution = self.interpolator(self.alpha, self.init_samples,
                                                                  self.target_sampler(self.init_samples.shape[0]),
                                                                  blur=self.ent_eps)
                    self.distribution_trace.append(np.copy(self.current_distribution))
                    self.alpha_trace.append(self.alpha)
                else:
                    self.current_distribution = self.target_sampler(self.init_samples.shape[0])
                print(f"Alpha: {self.alpha}")
        else:
            self.success_buffer.update(contexts, returns, self.target_sampler(self.init_samples.shape[0]))
            self.current_distribution = self.success_buffer.read_update()
            self.distribution_trace.append(np.copy(self.current_distribution))
            self.alpha_trace.append(self.alpha)

            if self.success_buffer.delta_reached:
                self.interpolation_started = True
                self.init_samples = self.current_distribution

    def sample(self):
        return self.current_distribution[np.random.randint(self.current_distribution.shape[0]), :]

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
