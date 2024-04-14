import os
import gym
import torch.nn
import numpy as np
from functools import partial
from stable_baselines3.common.vec_env import DummyVecEnv
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from deep_sprl.teachers.dummy_teachers import DiscreteSampler
from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
from deep_sprl.teachers.spl import SelfPacedWrapper, ExactGradient, ExactCurrOT


def logsumexp(x):
    xmax = np.max(x)
    return np.log(np.sum(np.exp(x - xmax))) + xmax


class PerformanceOracle:

    def __init__(self, fn):
        self.fn = fn

    def __call__(self):
        return self.fn()

    def update(self, *args, **kwargs):
        pass


class EMazeExperiment(AbstractExperiment):
    INITIAL_CONTEXTS = np.array([0, 1, 2, 20, 21, 22, 40, 41, 42], dtype=np.int64)
    TARGET_CONTEXTS = np.array([9, 10, 11, 29, 30, 31, 49, 50, 51], dtype=np.int64)
    N_CONTEXTS = 400

    def target_sampler(self, n=None, rng=None):
        if rng is None:
            rng = np.random

        idxs = rng.randint(0, self.TARGET_CONTEXTS.shape[0], size=(n,))
        return self.TARGET_CONTEXTS[idxs]

    DELTA = {Learner.PPO: 0.1}
    # Not required for exact currot
    METRIC_EPS = 0.2
    EP_PER_UPDATE = 20
    ENT_LB = {Learner.PPO: 0.}

    GRADIENT_DELTA = {Learner.PPO: 0.3}
    # Not required for exact gradient
    GRADIENT_EPS = 0.1
    GRADIENT_ENT = {Learner.PPO: 0.}

    STEPS_PER_ITER = 2048
    DISCOUNT_FACTOR = 0.99
    LAM = 0.99

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed):
        if "dist_fn" not in parameters:
            raise RuntimeError("Need to specify a distance function for this environment via the 'dist_fn' parameter")
        else:
            self.dist_fn = parameters["dist_fn"]
            if self.dist_fn == "perf":
                self.dist_arg = None
            else:
                self.dist_arg = self.dist_fn
            del parameters["dist_fn"]

        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed)
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)

    def create_environment(self, evaluation=False):
        env = gym.make("EMaze-v1", dist_fn=self.dist_arg)
        if evaluation or self.curriculum.default():
            probs = np.zeros(self.N_CONTEXTS)
            probs[self.TARGET_CONTEXTS] = 1. / self.TARGET_CONTEXTS.shape[0]
            teacher = DiscreteSampler(np.log(probs))
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False)
        elif self.curriculum.alp_gmm():
            raise RuntimeError("ALP-GMM is not applicable for this experiment")
        elif self.curriculum.goal_gan():
            raise RuntimeError("GoalGAN is not applicable for this experiment")
        elif self.curriculum.self_paced():
            raise RuntimeError("SPDL is not applicable for this experiment")
        elif self.curriculum.wasserstein():
            init_dist = np.zeros(self.N_CONTEXTS)
            init_dist[self.INITIAL_CONTEXTS] = 1
            init_dist /= np.sum(init_dist)

            mu = np.zeros(self.N_CONTEXTS)
            mu[self.TARGET_CONTEXTS] = 1
            mu /= np.sum(mu)

            # We initialize the performance model in the create_environment method since we need to learner for that
            teacher = ExactCurrOT(init_dist, env.unwrapped.dist_fn, None, self.DELTA[self.learner], mu,
                                  ent_lb=self.ENT_LB[self.learner])
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, episodes_per_update=self.EP_PER_UPDATE,
                                   context_visible=False)
        elif self.curriculum.gradient():
            init_dist = np.zeros(self.N_CONTEXTS)
            init_dist[self.INITIAL_CONTEXTS] = 1
            init_dist /= np.sum(init_dist)

            mu = np.zeros(self.N_CONTEXTS)
            mu[self.TARGET_CONTEXTS] = 1
            mu /= np.sum(mu)

            teacher = ExactGradient(init_dist, env.unwrapped.dist_fn, None,
                                    self.GRADIENT_DELTA[self.learner], mu, ent_eps=self.GRADIENT_ENT[self.learner])
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, episodes_per_update=self.EP_PER_UPDATE,
                                   context_visible=False)
        elif self.curriculum.acl():
            raise RuntimeError("ACL is not applicable for this experiment")
        elif self.curriculum.plr():
            raise RuntimeError("PLR is not applicable for this experiment")
        elif self.curriculum.vds():
            raise RuntimeError("VDS is not applicable for this experiment")
        elif self.curriculum.random():
            teacher = DiscreteSampler(np.log(np.ones(self.N_CONTEXTS) / self.N_CONTEXTS))
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False)
        else:
            raise RuntimeError("Invalid learning type")

        return env, DummyVecEnv([lambda: env])

    def create_learner_params(self):
        return dict(common=dict(gamma=self.DISCOUNT_FACTOR, seed=self.seed, verbose=0, device="cpu",
                                policy_kwargs=dict(net_arch=[128, 128, 128], activation_fn=torch.nn.ReLU)),
                    ppo=dict(n_steps=self.STEPS_PER_ITER, gae_lambda=self.LAM))

    def create_self_paced_teacher(self):
        raise RuntimeError("SPDL not supported in this env")

    def create_experiment(self):
        timesteps = 101 * self.STEPS_PER_ITER

        env, vec_env = self.create_environment(evaluation=False)
        model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

        if self.curriculum.wasserstein() or self.curriculum.gradient():
            perf_model = PerformanceOracle(partial(env.env.unwrapped.evaluate_agent, model, gamma=self.DISCOUNT_FACTOR))
            if self.dist_fn == "cur_perf":
                env.teacher.dist_fn.agent = model

            env.teacher.performance_model = perf_model

        callback_params = {"learner": interface, "env_wrapper": env, "save_interval": 5,
                           "step_divider": self.STEPS_PER_ITER}
        return model, timesteps, callback_params

    def get_env_name(self):
        return f"emaze_{self.dist_fn}"

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
