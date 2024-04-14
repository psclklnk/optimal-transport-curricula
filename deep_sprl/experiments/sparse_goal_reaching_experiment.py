import os
import gym
import torch
import numpy as np
from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from deep_sprl.teachers.spl import SelfPacedTeacherV2, SelfPacedWrapper, CurrOT, Gradient
from deep_sprl.teachers.dummy_teachers import UniformSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from deep_sprl.teachers.acl import ACL, ACLWrapper
from deep_sprl.teachers.plr import PLR, PLRWrapper, ValueFunction
from deep_sprl.teachers.vds import VDS, VDSWrapper
from deep_sprl.teachers.util import Subsampler
from deep_sprl.environments.sparse_goal_reaching import SparseGoalReachingEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3 import HerReplayBuffer


class DictExtractor:

    def __init__(self, obs_space, keys):
        self.keys = keys
        self.features_dim = 0
        for k in self.keys:
            self.features_dim += np.prod(obs_space.spaces[k].shape)

    def __call__(self, obs: dict):
        if isinstance(obs[self.keys[0]], np.ndarray):
            return np.concatenate([obs[k] for k in self.keys], axis=-1)
        else:
            return torch.cat([obs[k] for k in self.keys], dim=-1)


class MazeSampler:
    def __init__(self):
        self.LOWER_CONTEXT_BOUNDS = np.array([-9., -9., 0.05])
        self.UPPER_CONTEXT_BOUNDS = np.array([9., 9., 0.05])

    def sample(self):
        sample = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
        while not SparseGoalReachingEnv._is_feasible(sample):
            sample = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
        return sample

    def save(self, path):
        pass

    def load(self, path):
        pass


class SparseGoalReachingExperiment(AbstractExperiment):
    INITIAL_MEAN = np.array([0., 0., 10])
    INITIAL_VARIANCE = np.diag(np.square([4, 4, 5]))

    TARGET_LOWER_CONTEXT_BOUNDS = np.array([-9., -9., 0.05])
    TARGET_UPPER_CONTEXT_BOUNDS = np.array([9., 9., 0.05])

    LOWER_CONTEXT_BOUNDS = np.array([-9., -9., 0.05])
    UPPER_CONTEXT_BOUNDS = np.array([9., 9., 18.])

    DISCOUNT_FACTOR = 0.995

    DELTA = 0.8
    METRIC_EPS = 1.2
    KL_EPS = 0.25
    EP_PER_UPDATE = 50
    ENT_LB = 0.

    GRADIENT_EPS = {Learner.SAC: 0.05}
    GRADIENT_DELTA = {Learner.SAC: 0.6}
    GRADIENT_ENT = {Learner.SAC: 1e-2}

    # ACL Parameters [found after search over [0.05, 0.1, 0.2] x [0.01, 0.025, 0.05]]
    ACL_EPS = 0.2
    ACL_ETA = 0.05

    PLR_REPLAY_RATE = 0.55
    PLR_BUFFER_SIZE = 100
    PLR_BETA = 0.15
    PLR_RHO = 0.45

    VDS_NQ = 5
    VDS_LR = 5e-4
    VDS_EPOCHS = 10
    VDS_BATCHES = 80

    STEPS_PER_ITER = 10000
    LAM = 0.995
    ACHI_NET = dict(net_arch=[128, 128, 128], activation_fn=torch.nn.ReLU)

    AG_P_RAND = {Learner.SAC: 0.2}
    AG_FIT_RATE = {Learner.SAC: 200}
    AG_MAX_SIZE = {Learner.SAC: 500}

    GG_NOISE_LEVEL = {Learner.SAC: 0.1}
    GG_FIT_RATE = {Learner.SAC: 200}
    GG_P_OLD = {Learner.SAC: 0.2}

    def target_log_likelihood(self, cs):
        norm = np.prod(self.UPPER_CONTEXT_BOUNDS[:2] - self.LOWER_CONTEXT_BOUNDS[:2]) * (0.01 * 1 + 17.94 * 1e-4)
        return np.where(cs[:, -1] < 0.06, np.log(1 / norm) * np.ones(cs.shape[0]),
                        np.log(1e-4 / norm) * np.ones(cs.shape[0]))

    def target_sampler(self, n=None, rng=None):
        if n is None:
            n = self.EP_PER_UPDATE

        if rng is None:
            rng = np.random
        return rng.uniform(self.TARGET_LOWER_CONTEXT_BOUNDS, self.TARGET_UPPER_CONTEXT_BOUNDS, size=(n, 3))

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed):
        if "HER_REPLAYS" in parameters:
            self.her_replays = int(parameters["HER_REPLAYS"])
            del parameters["HER_REPLAYS"]
        else:
            self.her_replays = -1

        if "ONLINE_HER" in parameters:
            self.online_her = parameters["ONLINE_HER"].lower() == "true"
            del parameters["ONLINE_HER"]
        else:
            self.online_her = True

        if "TARGET_SLICE" in parameters:
            self.two_d = bool(parameters["TARGET_SLICE"])
            del parameters["TARGET_SLICE"]
            self.UPPER_CONTEXT_BOUNDS = np.copy(self.TARGET_UPPER_CONTEXT_BOUNDS)
        else:
            self.two_d = False

        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed)
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)

    def create_environment(self, evaluation=False):
        env = gym.make("SparseGoalReaching-v1", dict_observation=self.her_replays >= 0)
        if evaluation or self.curriculum.default():
            teacher = MazeSampler()
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True, reward_from_info=True)
        elif self.curriculum.alp_gmm():
            teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                             fit_rate=self.AG_FIT_RATE[self.learner], random_task_ratio=self.AG_P_RAND[self.learner],
                             max_size=self.AG_MAX_SIZE[self.learner])
            env = ALPGMMWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.goal_gan():
            init_samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS,
                                             size=(1000, 3))
            teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                              state_noise_level=self.GG_NOISE_LEVEL[self.learner], success_distance_threshold=0.01,
                              update_size=self.GG_FIT_RATE[self.learner], n_rollouts=4, goid_lb=0.25, goid_ub=0.75,
                              p_old=self.GG_P_OLD[self.learner], pretrain_samples=init_samples)
            env = GoalGANWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.self_paced() or self.curriculum.wasserstein():
            teacher = self.create_self_paced_teacher(with_callback=False)
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True, reward_from_info=True,
                                   use_undiscounted_reward=True, episodes_per_update=self.EP_PER_UPDATE)
        elif self.curriculum.gradient():
            bounds = (self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
            init_samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(500, 3))
            teacher = Gradient(bounds, init_samples, self.target_sampler, self.GRADIENT_EPS[self.learner],
                               self.GRADIENT_DELTA[self.learner], self.GRADIENT_ENT[self.learner])
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, episodes_per_update=self.EP_PER_UPDATE,
                                   reward_from_info=True, use_undiscounted_reward=True, context_visible=True)
        elif self.curriculum.acl():
            bins = 20
            teacher = ACL(bins * bins * bins, self.ACL_ETA, eps=self.ACL_EPS)
            env = ACLWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                             context_post_processing=Subsampler(self.LOWER_CONTEXT_BOUNDS.copy(),
                                                                self.UPPER_CONTEXT_BOUNDS.copy(),
                                                                [bins, bins, bins]))
        elif self.curriculum.plr():
            teacher = PLR(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.PLR_REPLAY_RATE,
                          self.PLR_BUFFER_SIZE, self.PLR_BETA, self.PLR_RHO)
            env = PLRWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                             value_fn=ValueFunction(env.observation_space.shape[0] + self.LOWER_CONTEXT_BOUNDS.shape[0],
                                                    [128, 128, 128], torch.nn.ReLU(),
                                                    {"steps_per_iter": 2048, "noptepochs": 10,
                                                     "minibatches": 32, "lr": 3e-4}), lam=self.LAM)
        elif self.curriculum.vds():
            teacher = VDS(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.DISCOUNT_FACTOR, self.VDS_NQ,
                          q_train_config={"replay_size": 5 * self.STEPS_PER_ITER, "lr": self.VDS_LR,
                                          "n_epochs": self.VDS_EPOCHS, "batches_per_epoch": self.VDS_BATCHES,
                                          "steps_per_update": self.STEPS_PER_ITER})
            env = VDSWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.random():
            teacher = UniformSampler(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True, reward_from_info=True)
        else:
            raise RuntimeError("Invalid learning type")

        return env, env

    def create_learner_params(self):
        learner_params = dict(common=dict(gamma=self.DISCOUNT_FACTOR, seed=self.seed, verbose=0,
                                          policy_kwargs=self.ACHI_NET, device="cpu"),
                              sac=dict(learning_rate=3e-4, buffer_size=200000, learning_starts=1000, batch_size=512,
                                       train_freq=5, target_entropy="auto"))

        if self.her_replays >= 0:
            learner_params["common"]["policy_kwargs"]["features_extractor_class"] = DictExtractor
            learner_params["common"]["policy_kwargs"]["features_extractor_kwargs"] = {"keys": ["observation",
                                                                                               "desired_goal"]}
            learner_params["sac"]["replay_buffer_class"] = HerReplayBuffer
            learner_params["sac"]["replay_buffer_kwargs"] = dict(n_sampled_goal=self.her_replays,
                                                                 goal_selection_strategy=GoalSelectionStrategy.FUTURE,
                                                                 online_sampling=self.online_her)

        return learner_params

    def create_experiment(self):
        timesteps = 400 * self.STEPS_PER_ITER

        env, vec_env = self.create_environment(evaluation=False)
        model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

        if isinstance(env, PLRWrapper):
            env.learner = interface

        if isinstance(env, VDSWrapper):
            env.teacher.initialize_teacher(env, interface,
                                           lambda contexts: np.concatenate(
                                               (
                                                   SparseGoalReachingEnv.sample_initial_state(contexts.shape[0]),
                                                   contexts),
                                               axis=-1))

        callback_params = {"learner": interface, "env_wrapper": env, "save_interval": 10,
                           "step_divider": self.STEPS_PER_ITER}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self, with_callback=False):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced():
            return SelfPacedTeacherV2(self.target_log_likelihood, self.target_sampler, self.INITIAL_MEAN.copy(),
                                      self.INITIAL_VARIANCE.copy(), bounds, self.DELTA, max_kl=self.KL_EPS,
                                      std_lower_bound=None, kl_threshold=None)
        elif self.curriculum.wasserstein():
            init_samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(500, 3))
            return CurrOT(bounds, init_samples, self.target_sampler, self.DELTA, self.METRIC_EPS,
                          wait_until_threshold=True)
        else:
            raise RuntimeError('Invalid self-paced curriculum type')

    def get_env_name(self):
        suffix = f"{'_target_slice' if self.two_d else ''}"

        if self.her_replays < 0:
            return f"sparse_goal_reaching{suffix}"
        else:
            return f"sparse_goal_reaching_her_{self.her_replays}{suffix}{'' if self.online_her else '_offline'}"

    def evaluate_learner(self, path, render=False):
        model_load_path = os.path.join(path, "model.zip")
        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env)
        for i in range(0, 200):
            obs = self.vec_eval_env.reset()
            done = False
            while not done:
                action = model.step(obs, state=None, deterministic=False)
                obs, rewards, done, infos = self.vec_eval_env.step(action)
                if render:
                    self.vec_eval_env.render(mode="human")

        stats = self.eval_env.get_statistics()
        return stats[0]
