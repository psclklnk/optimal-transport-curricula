import os
import gym
import torch.nn
import numpy as np
from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
from deep_sprl.teachers.spl import SelfPacedWrapper, CurrOT, Gradient
from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from deep_sprl.teachers.dummy_teachers import DistributionSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from deep_sprl.teachers.plr import PLRWrapper
from deep_sprl.teachers.vds import VDSWrapper


def reduce_context(contexts):
    context_dim = contexts.shape[1] // 3
    helper = np.arange(contexts.shape[0])
    max_abs_pos = np.argmax(np.abs(contexts[:, :context_dim]), axis=-1)
    return np.stack([contexts[helper, max_abs_pos],
                     np.min(contexts[helper, context_dim:2 * context_dim], axis=-1),
                     np.min(contexts[helper, 2 * context_dim:], axis=-1)], axis=-1)


def extend_context(contexts, reduced_context_bounds, ext_dim, min_reduction: bool = True):
    pos_ub = np.abs(contexts[:, 0])
    ext_pos = np.sign(np.random.uniform(-1., 1., size=(contexts.shape[0], ext_dim))) * \
              np.random.uniform(np.zeros((contexts.shape[0], ext_dim)), pos_ub[:, None].repeat(ext_dim, axis=-1))
    if min_reduction:
        ext_widths1 = np.random.uniform(contexts[:, 1][:, None].repeat(ext_dim, axis=-1),
                                        reduced_context_bounds[1][1] * np.ones((contexts.shape[0], ext_dim)))
        ext_widths2 = np.random.uniform(contexts[:, 2][:, None].repeat(ext_dim, axis=-1),
                                        reduced_context_bounds[1][2] * np.ones((contexts.shape[0], ext_dim)))
    else:
        ext_widths1 = np.random.uniform(reduced_context_bounds[0][1] * np.ones((contexts.shape[0], ext_dim)),
                                        contexts[:, 1][:, None].repeat(ext_dim, axis=-1))
        ext_widths2 = np.random.uniform(reduced_context_bounds[0][2] * np.ones((contexts.shape[0], ext_dim)),
                                        contexts[:, 2][:, None].repeat(ext_dim, axis=-1))

    full_pos = np.concatenate((contexts[:, [0]], ext_pos), axis=1)
    full_width1 = np.concatenate((contexts[:, [1]], ext_widths1), axis=1)
    full_width2 = np.concatenate((contexts[:, [2]], ext_widths2), axis=1)

    # Switch the context arbitrarily
    helper = np.arange(contexts.shape[0])
    idx1 = np.random.randint(0, full_pos.shape[1], size=helper.shape)
    tmp = np.copy(full_pos[helper, 0])
    full_pos[helper, 0] = full_pos[helper, idx1]
    full_pos[helper, idx1] = tmp

    idx1 = np.random.randint(0, full_width1.shape[1], size=helper.shape)
    tmp = np.copy(full_width1[helper, 0])
    full_width1[helper, 0] = full_width1[helper, idx1]
    full_width1[helper, idx1] = tmp

    idx1 = np.random.randint(0, full_width2.shape[1], size=helper.shape)
    tmp = np.copy(full_width2[helper, 0])
    full_width2[helper, 0] = full_width2[helper, idx1]
    full_width2[helper, idx1] = tmp

    return np.concatenate((full_pos, full_width1, full_width2), axis=1)


class PointMassNDExperiment(AbstractExperiment):
    LOWER_CONTEXT_BOUNDS = None
    UPPER_CONTEXT_BOUNDS = None

    def easy_context_sampler(self, n=None, rng=None):
        if n is None:
            n = self.EP_PER_UPDATE

        if rng is None:
            rng = np.random

        # We first sample low dimensional contexts uniformly and then project them back up
        bounds = (reduce_context(self.LOWER_CONTEXT_BOUNDS[None, :])[0],
                  reduce_context(self.UPPER_CONTEXT_BOUNDS[None, :])[0])
        contexts = rng.uniform(self.LOWER_CONTEXT_BOUNDS[0::self._dim],
                               self.UPPER_CONTEXT_BOUNDS[0::self._dim], size=(n, 3))
        return extend_context(contexts, bounds, self._dim - 1,
                              min_reduction=not self._max_reduction)

    def slice_target_sampler(self, n=None, rng=None):
        if n is None:
            n = self.EP_PER_UPDATE

        if rng is None:
            rng = np.random

        TARGET_MEANS = np.array([[3.] + [0.26] * 2, [-3.] + [0.26] * 2])
        TARGET_VARIANCES = np.array([np.diag([1e-6] * 3),
                                     np.diag([1e-6] * 3)])

        if n % 2 == 0:
            s0 = rng.multivariate_normal(TARGET_MEANS[0], TARGET_VARIANCES[0], size=n // 2)
            s1 = rng.multivariate_normal(TARGET_MEANS[1], TARGET_VARIANCES[1], size=n // 2)
        else:
            if np.random.uniform(0, 1) > 0.5:
                s0 = rng.multivariate_normal(TARGET_MEANS[0], TARGET_VARIANCES[0], size=(n - 1) // 2)
                s1 = rng.multivariate_normal(TARGET_MEANS[1], TARGET_VARIANCES[1], size=(n + 1) // 2)
            else:
                s0 = rng.multivariate_normal(TARGET_MEANS[0], TARGET_VARIANCES[0], size=(n + 1) // 2)
                s1 = rng.multivariate_normal(TARGET_MEANS[1], TARGET_VARIANCES[1], size=(n - 1) // 2)

        # Project the lower-dimensional samples to equivalent high-dimensional ones
        if self._dim > 1:
            reduced_context_bounds = (self.LOWER_CONTEXT_BOUNDS[0::self._dim],
                                      self.UPPER_CONTEXT_BOUNDS[0::self._dim])
            return extend_context(np.concatenate((s0, s1), axis=0), reduced_context_bounds, ext_dim=self._dim - 1,
                                  min_reduction=not self._max_reduction)
        else:
            return np.concatenate((s0, s1), axis=0)

    def point_target_sampler(self, n=None, rng=None):
        if n is None:
            n = self.EP_PER_UPDATE

        if rng is None:
            rng = np.random

        TARGET_MEANS = np.array([[3.] * self._dim + [0.26] * 2 * self._dim,
                                 [-3.] * self._dim + [0.26] * 2 * self._dim])
        TARGET_VARIANCES = np.array([np.diag([1e-6] * 3 * self._dim),
                                     np.diag([1e-6] * 3 * self._dim)])

        if n % 2 == 0:
            s0 = rng.multivariate_normal(TARGET_MEANS[0], TARGET_VARIANCES[0], size=n // 2)
            s1 = rng.multivariate_normal(TARGET_MEANS[1], TARGET_VARIANCES[1], size=n // 2)
        else:
            if np.random.uniform(0, 1) > 0.5:
                s0 = rng.multivariate_normal(TARGET_MEANS[0], TARGET_VARIANCES[0], size=(n - 1) // 2)
                s1 = rng.multivariate_normal(TARGET_MEANS[1], TARGET_VARIANCES[1], size=(n + 1) // 2)
            else:
                s0 = rng.multivariate_normal(TARGET_MEANS[0], TARGET_VARIANCES[0], size=(n + 1) // 2)
                s1 = rng.multivariate_normal(TARGET_MEANS[1], TARGET_VARIANCES[1], size=(n - 1) // 2)

        return np.concatenate((s0, s1), axis=0)

    INITIAL_MEAN = None
    INITIAL_VARIANCE = None

    STD_LOWER_BOUND = None
    KL_THRESHOLD = None
    KL_EPS = None
    DELTA = 4.0
    METRIC_EPS = None
    EP_PER_UPDATE = 20
    # Not used in the continuous implementation
    ENT_LB = 0.

    GRADIENT_EPS = {Learner.PPO: 0.2}
    GRADIENT_DELTA = {Learner.PPO: 3.0}
    GRADIENT_ENT = {Learner.PPO: 1e-2}

    STEPS_PER_ITER = 4096
    DISCOUNT_FACTOR = 0.95
    LAM = 0.99

    # ACL Parameters [found after search over [0.05, 0.1, 0.2] x [0.01, 0.025, 0.05]]
    ACL_EPS = None
    ACL_ETA = None

    PLR_REPLAY_RATE = None
    PLR_BUFFER_SIZE = None
    PLR_BETA = None
    PLR_RHO = None

    VDS_NQ = None
    VDS_LR = None
    VDS_EPOCHS = None
    VDS_BATCHES = None

    AG_P_RAND = {Learner.PPO: 0.1}
    AG_FIT_RATE = {Learner.PPO: 100}
    AG_MAX_SIZE = {Learner.PPO: 500}

    GG_NOISE_LEVEL = {Learner.PPO: None}
    GG_FIT_RATE = {Learner.PPO: None}
    GG_P_OLD = {Learner.PPO: None}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed):
        if "DIM" not in parameters:
            raise RuntimeError("Environment requires 'DIM' parameter to be specified")
        else:
            dim = int(parameters["DIM"])
            del parameters["DIM"]

        if "MAX" in parameters:
            self._max_reduction = bool(parameters["MAX"])
            del parameters["MAX"]
        else:
            self._max_reduction = False

        if "BETTER_INIT" in parameters:
            self._better_init = bool(parameters["BETTER_INIT"])
            del parameters["BETTER_INIT"]
        else:
            self._better_init = False

        if "TARGET_SLICES" in parameters:
            self._slices = bool(parameters["TARGET_SLICES"])
            del parameters["TARGET_SLICES"]
        else:
            self._slices = False

        self._dim = dim

        # Set the boundaries etc
        self.LOWER_CONTEXT_BOUNDS = np.array([-4.] * self._dim + [0.25] * 2 * self._dim)
        self.UPPER_CONTEXT_BOUNDS = np.array([4.] * self._dim + [4.] * 2 * self._dim)

        # We do this to get the same distances in each run. The experiment seed will be set later
        old_state = np.random.get_state()

        np.random.seed(0)
        if self._slices:
            target_sampler = self.slice_target_sampler
        else:
            target_sampler = self.point_target_sampler
        target_samples = target_sampler(100)
        target_samples2 = target_sampler(100)
        dists = np.sum(np.square(target_samples[:, None, :] - target_samples2[None, ...]), axis=-1)
        from scipy.optimize import linear_sum_assignment
        idxs1, idxs2 = linear_sum_assignment(dists)
        eps = np.sqrt(np.mean(dists[idxs1, idxs2]))
        np.random.set_state(old_state)

        self.METRIC_EPS = np.round(max(1.2 * eps,
                                       0.05 * np.linalg.norm(self.LOWER_CONTEXT_BOUNDS - self.UPPER_CONTEXT_BOUNDS)), 2)

        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed)
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)

    def create_environment(self, evaluation=False):
        env = gym.make("HighDimPointMass-v1", dim=self._dim, max_reduction=self._max_reduction)

        if evaluation or self.curriculum.default():
            if self._slices:
                target_sampler = self.slice_target_sampler
            else:
                target_sampler = self.point_target_sampler
            teacher = DistributionSampler(target_sampler, self.LOWER_CONTEXT_BOUNDS,
                                          self.UPPER_CONTEXT_BOUNDS)
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False)
        elif self.curriculum.alp_gmm():
            teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                             fit_rate=self.AG_FIT_RATE[self.learner], random_task_ratio=self.AG_P_RAND[self.learner],
                             max_size=self.AG_MAX_SIZE[self.learner])
            env = ALPGMMWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False)
        elif self.curriculum.goal_gan():
            raise NotImplementedError("For this experiment, we only investigated the scalability of CurrOT,"
                                      "ALP-GMM, and Gradient")
        elif self.curriculum.self_paced() or self.curriculum.wasserstein():
            teacher = self.create_self_paced_teacher(with_callback=False)
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, episodes_per_update=self.EP_PER_UPDATE,
                                   context_visible=False)
        elif self.curriculum.gradient():
            teacher = self.create_gradient_teacher()
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, episodes_per_update=self.EP_PER_UPDATE,
                                   context_visible=False)
        elif self.curriculum.acl():
            raise NotImplementedError("For this experiment, we only investigated the scalability of CurrOT,"
                                      "ALP-GMM, and Gradient")
        elif self.curriculum.plr():
            raise NotImplementedError("For this experiment, we only investigated the scalability of CurrOT,"
                                      "ALP-GMM, and Gradient")
        elif self.curriculum.vds():
            raise NotImplementedError("For this experiment, we only investigated the scalability of CurrOT,"
                                      "ALP-GMM, and Gradient")
        elif self.curriculum.random():
            raise NotImplementedError("For this experiment, we only investigated the scalability of CurrOT,"
                                      "ALP-GMM, and Gradient")
        else:
            raise RuntimeError("Invalid learning type")

        return env, DummyVecEnv([lambda: env])

    def create_learner_params(self):
        return dict(common=dict(gamma=self.DISCOUNT_FACTOR, seed=self.seed, verbose=0, device="cpu",
                                policy_kwargs=dict(net_arch=[128, 128, 128], activation_fn=torch.nn.Tanh)),
                    ppo=dict(n_steps=self.STEPS_PER_ITER, gae_lambda=self.LAM, batch_size=128))

    def create_experiment(self):
        timesteps = 200 * self.STEPS_PER_ITER

        env, vec_env = self.create_environment(evaluation=False)
        model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

        if isinstance(env, PLRWrapper):
            env.learner = interface

        if isinstance(env, VDSWrapper):
            state_provider = lambda contexts: np.concatenate(
                [np.repeat(np.array([0., 0., -3., 0.])[None, :], contexts.shape[0], axis=0),
                 contexts], axis=-1)
            env.teacher.initialize_teacher(env, interface, state_provider)

        callback_params = {"learner": interface, "env_wrapper": env, "save_interval": 5,
                           "step_divider": self.STEPS_PER_ITER}
        return model, timesteps, callback_params

    def get_init_and_target(self):
        if self._better_init:
            init_samples = self.easy_context_sampler(n=100)
        else:
            init_samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS,
                                             size=(100, 3 * self._dim))

        if self._slices:
            target_sampler = self.slice_target_sampler
        else:
            target_sampler = self.point_target_sampler

        return init_samples, target_sampler

    def create_gradient_teacher(self):
        bounds = (self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)

        init_samples, target_sampler = self.get_init_and_target()
        return Gradient(bounds, init_samples, target_sampler, self.GRADIENT_EPS[self.learner],
                        self.GRADIENT_DELTA[self.learner], self.GRADIENT_ENT[self.learner])

    def create_self_paced_teacher(self, with_callback=False):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced():
            raise NotImplementedError("For this experiment, we only investigated the scalability of CurrOT"
                                      "and Gradient")
        else:
            init_samples, target_sampler = self.get_init_and_target()
            currot = CurrOT(bounds, init_samples, target_sampler, self.DELTA, self.METRIC_EPS,
                            wait_until_threshold=True)
            return currot

    def get_env_name(self):
        return f"point_mass_{3 * self._dim}d{'_slice' if self._slices else ''}" \
               f"{'_better_init' if self._better_init else ''}{'_max' if self._max_reduction else '_min'}"

    def evaluate_learner(self, path):
        model_load_path = os.path.join(path, "model.zip")
        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env)
        for i in range(0, 100):
            obs = self.vec_eval_env.reset()
            done = False
            while not done:
                action = model.step(obs, state=None, deterministic=False)
                obs, rewards, done, infos = self.vec_eval_env.step(action)

        return self.eval_env.get_statistics()[1]
