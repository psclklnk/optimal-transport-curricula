import copy
import numpy as np
from TeachMyAgent.teachers.algos.AbstractTeacher import AbstractTeacher
from .currot import CurrOT, Gradient


class TMACurrot(AbstractTeacher):

    def __init__(self, context_lb, context_ub, env_reward_lb, env_reward_ub, perf_lb=180, n_samples=500,
                 episodes_per_update=50, epsilon=None, callback=None, seed=None):

        super().__init__(context_lb, context_ub, env_reward_lb, env_reward_ub, seed=seed)

        if epsilon is None:
            epsilon = 0.05 * np.linalg.norm(np.array(context_ub) - np.array(context_lb))

        if perf_lb is None:
            perf_lb = 0.5 * (env_reward_ub - env_reward_lb) + env_reward_lb

        if episodes_per_update is None:
            episodes_per_update = 0.25 * n_samples
        self.episodes_per_update = episodes_per_update

        print(f"Running with epsilon {epsilon}, perf_lb {perf_lb}, ep_per_update {self.episodes_per_update}")

        # Create an array if we use the same number of bins per dimension
        target_sampler = lambda n: np.random.uniform(context_lb, context_ub, size=(n, len(context_lb)))
        init_samples = np.random.uniform(context_lb, context_ub, size=(n_samples, len(context_lb)))

        self.curriculum = CurrOT((np.array(context_lb), np.array(context_ub)),
                                 init_samples, target_sampler, perf_lb, epsilon,
                                 wait_until_threshold=False)

        self.context_buffer = []
        self.return_buffer = []
        self.bk = {"teacher_snapshots": []}

    def episodic_update(self, task, reward, is_success):

        # self.sampler.update(self.sample_idx, reward)
        self.curriculum.on_rollout_end(task, reward)
        self.context_buffer.append(task)
        self.return_buffer.append(reward)
        # print("Updated task %d" % self.sample_idx)

        if len(self.context_buffer) >= self.episodes_per_update:
            new_snapshot = {"context_buffer": copy.deepcopy(self.context_buffer),
                            "return_buffer": copy.deepcopy(self.return_buffer)}
            contexts = np.array(self.context_buffer)
            returns = np.array(self.return_buffer)
            self.context_buffer.clear()
            self.return_buffer.clear()
            self.curriculum.update_distribution(contexts.astype(np.float64), returns.astype(np.float64))
            new_snapshot["current_samples"] = copy.deepcopy(self.curriculum.teacher.current_samples)
            new_snapshot["success_buffer"] = copy.deepcopy(self.curriculum.success_buffer.contexts)
            self.bk["teacher_snapshots"].append(new_snapshot)

    def sample_task(self):
        return self.curriculum.sample().astype(np.float32)

    def is_non_exploratory_task_sampling_available(self):
        return True

    def non_exploratory_task_sampling(self):
        task_idx = np.random.randint(0, self.curriculum.teacher.current_samples.shape[0])
        task = np.clip(self.curriculum.teacher.current_samples[task_idx, :], self.curriculum.context_bounds[0],
                       self.curriculum.context_bounds[1]).astype(np.float32)
        return {"task": task, "infos": None}

    def save(self, path):
        self.curriculum.save(path)

    def load(self, path):
        self.curriculum.load(path)


class TMAGradient(AbstractTeacher):

    def __init__(self, context_lb, context_ub, env_reward_lb, env_reward_ub, perf_lb=180, n_samples=500,
                 episodes_per_update=50, optimize_initial_samples=False, epsilon=None, callback=None, seed=None):

        super().__init__(context_lb, context_ub, env_reward_lb, env_reward_ub, seed=seed)

        if epsilon is None:
            epsilon = 0.05

        if perf_lb is None:
            perf_lb = 0.5 * (env_reward_ub - env_reward_lb) + env_reward_lb

        if episodes_per_update is None:
            episodes_per_update = 0.25 * n_samples
        self.episodes_per_update = episodes_per_update

        print(f"Running with epsilon {epsilon}, perf_lb {perf_lb}, ep_per_update {self.episodes_per_update}")

        # Create an array if we use the same number of bins per dimension
        target_sampler = lambda n: np.random.uniform(context_lb, context_ub, size=(n, len(context_lb)))
        init_samples = np.random.uniform(context_lb, context_ub, size=(n_samples, len(context_lb)))

        self.curriculum = Gradient((np.array(context_lb), np.array(context_ub)),
                                   init_samples, target_sampler, epsilon, perf_lb,
                                   optimize_initial_samples=optimize_initial_samples)

        self.context_buffer = []
        self.return_buffer = []
        self.bk = {"teacher_snapshots": []}

    def episodic_update(self, task, reward, is_success):

        # self.sampler.update(self.sample_idx, reward)
        self.context_buffer.append(task)
        self.return_buffer.append(reward)
        # print("Updated task %d" % self.sample_idx)

        if len(self.context_buffer) >= self.episodes_per_update:
            new_snapshot = {"context_buffer": copy.deepcopy(self.context_buffer),
                            "return_buffer": copy.deepcopy(self.return_buffer)}
            contexts = np.array(self.context_buffer)
            returns = np.array(self.return_buffer)
            self.context_buffer.clear()
            self.return_buffer.clear()
            self.curriculum.update_distribution(contexts.astype(np.float64), returns.astype(np.float64))
            new_snapshot["current_samples"] = copy.deepcopy(self.curriculum.current_distribution)
            if self.curriculum.success_buffer is not None:
                new_snapshot["success_buffer"] = copy.deepcopy(self.curriculum.success_buffer.contexts)
            self.bk["teacher_snapshots"].append(new_snapshot)

    def sample_task(self):
        return self.curriculum.sample().astype(np.float32)

    def is_non_exploratory_task_sampling_available(self):
        return True

    def non_exploratory_task_sampling(self):
        task_idx = np.random.randint(0, self.curriculum.current_distribution.shape[0])
        task = np.clip(self.curriculum.current_distribution[task_idx, :], self.mins, self.maxs).astype(np.float32)
        return {"task": task, "infos": None}

    def save(self, path):
        self.curriculum.save(path)

    def load(self, path):
        self.curriculum.load(path)