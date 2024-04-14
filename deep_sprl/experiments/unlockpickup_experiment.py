import os
import pickle

import gym
import torch.nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import product
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from deep_sprl.experiments.abstract_experiment import AbstractExperiment
from deep_sprl.teachers.spl import SelfPacedWrapper, DiscreteCurrOT, DiscreteGradient
from deep_sprl.teachers.dummy_teachers import UniformDiscreteSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from deep_sprl.teachers.acl import ACL, ACLWrapper
from deep_sprl.teachers.plr import PLR, PLRWrapper
from deep_sprl.teachers.vds import VDS, VDSWrapper
from deep_sprl.environments.unlockpickup.unlock_pickup_context import UnlockPickupContext, get_context_distance, \
    NeighbourOracle


class UnlockPickupFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim=64, history_size: int = 1):
        super().__init__(observation_space, features_dim=cnn_output_dim)
        self.history_size = history_size

        self.image_branch = torch.nn.Sequential(
            torch.nn.Conv2d(3 * self.history_size, int(0.5 * cnn_output_dim), (2, 2)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(int(0.5 * cnn_output_dim), int(0.5 * cnn_output_dim), (2, 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(int(0.5 * cnn_output_dim), cnn_output_dim, (2, 2)),
            torch.nn.ReLU()
        )

    def __call__(self, obs):
        image = torch.moveaxis(torch.reshape(obs, obs.shape[:-1] + (7, 7, 3)), -1, -3)
        encoded_image = self.image_branch(torch.flatten(image, 1, 2))
        assert encoded_image.shape[2] == 1
        assert encoded_image.shape[3] == 1
        return torch.flatten(encoded_image, start_dim=1)


class VDSFeatureExtractor(torch.nn.Module):

    def __init__(self, cnn_output_dim: int, n_actions: int, action_embedding_dim: int, history_size: int = 1):
        super().__init__()
        self.state_encoder = UnlockPickupFeatureExtractor(None, cnn_output_dim, history_size=history_size)
        self.action_encoder = torch.nn.Embedding(n_actions, action_embedding_dim)

        self.feature_dim = self.state_encoder.features_dim + action_embedding_dim

    def __call__(self, states, actions):
        batch_dim = states.shape[:-2]
        encoded_states = torch.reshape(self.state_encoder(torch.flatten(states, 0, -3)), batch_dim + (-1,))

        return torch.cat((encoded_states, self.action_encoder(actions.flatten(len(batch_dim) - 1).long())), dim=-1)


class UnlockPickupExperiment(AbstractExperiment):

    def target_sampler(self, n=None, rng=None):
        if rng is None:
            rng = np.random

        assert n % self.TARGET_SAMPLES.shape[0] == 0
        n_sub = n // self.TARGET_SAMPLES.shape[0]

        sub_indices = rng.randint(0, self.TARGET_SAMPLES.shape[1], size=(self.TARGET_SAMPLES.shape[0], n_sub))
        samples = self.TARGET_SAMPLES[np.arange(self.TARGET_SAMPLES.shape[0])[:, None], sub_indices]
        return np.reshape(samples, (-1, samples.shape[-1]))

    def neighbour_oracle(self):
        pass

    ALL_CONTEXTS = None
    INIT_SAMPLES = None
    TARGET_SAMPLES = None
    INITIAL_STATES = None

    DELTA = 0.6
    METRIC_EPS = 3.
    EP_PER_UPDATE = 50
    ENT_LB = 0.

    GRADIENT_DELTA = 0.6
    GRADIENT_EPS = 0.05
    GRADIENT_ENT = 0.

    STEPS_PER_ITER = 10000
    DISCOUNT_FACTOR = 0.99
    LAM = 0.99

    # ACL Parameters [found after search over [0.05, 0.1, 0.2] x [0.01, 0.025, 0.05]]
    ACL_EPS = 0.1
    ACL_ETA = 0.025

    PLR_REPLAY_RATE = 0.55
    PLR_BUFFER_SIZE = 2000
    PLR_BETA = 0.45
    PLR_RHO = 0.45

    VDS_NQ = 5
    VDS_LR = 1e-3
    VDS_EPOCHS = 5
    VDS_BATCHES = 20

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed):
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed)
        self._init_samples()
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)
        self.distance_function = get_context_distance()
        neighbour_file = Path(__file__).resolve().parent / "neighbours.pkl"
        if neighbour_file.exists():
            self.neighbour_oracle = NeighbourOracle.load(Path(__file__).resolve().parent / "neighbours.pkl")
        else:
            self.neighbour_oracle = NeighbourOracle(self.ALL_CONTEXTS, self.distance_function, self.METRIC_EPS)
            self.neighbour_oracle.save(neighbour_file)

    def _init_samples(self):
        # Generate the initial samples
        init_samples = []
        target_samples = []
        for door_y_pos in [1, 2, 3, 4]:
            for box_pos in product(range(6, 10), range(1, 5)):
                if box_pos[0] == 6 and box_pos[1] == door_y_pos:
                    agent_pos = (5, door_y_pos)
                    key_pos = (4, door_y_pos)
                else:
                    if door_y_pos == box_pos[1]:
                        agent_pos = box_pos[0] - 1, door_y_pos
                        key_pos = agent_pos[0], door_y_pos + (1 if door_y_pos < 4 else -1)
                    else:
                        agent_pos = box_pos[0], box_pos[1] + (-1 if door_y_pos < box_pos[1] else 1)
                        key_pos = box_pos[0] + (1 if box_pos[0] < 9 else -1), agent_pos[1]

                init_context = UnlockPickupContext.from_desc(agent_pos, door_y_pos, key_pos, box_pos, door_open=True)
                init_samples.append(init_context.to_array())

                current_target_samples = []
                for agent_pos in product(range(1, 5), range(1, 5)):
                    for key_pos in product(range(1, 5), range(1, 5)):
                        if agent_pos != key_pos:
                            target_context = UnlockPickupContext.from_desc(agent_pos, door_y_pos, key_pos, box_pos,
                                                                           door_open=False)
                            current_target_samples.append(target_context.to_array())
                target_samples.append(current_target_samples)

        self.INIT_SAMPLES = np.array(init_samples)
        self.TARGET_SAMPLES = np.array(target_samples)
        all_contexts = UnlockPickupContext.get_array(np.arange(UnlockPickupContext.N_CONTEXTS))
        self.ALL_CONTEXTS = all_contexts[UnlockPickupContext.is_valid(all_contexts)]

        # Finally, we populate the states
        init_states_path = Path(__file__).resolve().parent / "initial_states.pkl"
        if init_states_path.exists():
            with open(init_states_path, "rb") as f:
                ref_contexts, self.INITIAL_STATES = pickle.load(f)
            assert np.all(ref_contexts == self.ALL_CONTEXTS)
        else:
            self.INITIAL_STATES = []
            env = gym.make("UnlockPickup-v1")
            for context in tqdm(self.ALL_CONTEXTS):
                env.unwrapped.context = UnlockPickupContext(context)
                self.INITIAL_STATES.append(env.reset())

            with open(init_states_path, "wb") as f:
                pickle.dump((self.ALL_CONTEXTS, self.INITIAL_STATES), f)

        self.INITIAL_STATES = np.array(self.INITIAL_STATES)

    def create_environment(self, evaluation=False):
        env = gym.make("UnlockPickup-v1")
        if evaluation or self.curriculum.default():
            teacher = UniformDiscreteSampler(self.TARGET_SAMPLES)
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False,
                              context_post_processing=UnlockPickupContext)
        elif self.curriculum.alp_gmm():
            raise RuntimeError("ALP-GMM not supported for this environment")
        elif self.curriculum.goal_gan():
            raise RuntimeError("GoalGAN not supported for this environment")
        elif self.curriculum.gradient():
            teacher = DiscreteGradient(np.repeat(self.INIT_SAMPLES.copy(), 10, axis=0), self.target_sampler,
                                       self.GRADIENT_DELTA, self.GRADIENT_EPS, self.distance_function,
                                       self.ALL_CONTEXTS)
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, episodes_per_update=self.EP_PER_UPDATE,
                                   context_visible=False, use_undiscounted_reward=False,
                                   context_post_processing=UnlockPickupContext)
        elif self.curriculum.self_paced() or self.curriculum.wasserstein():
            teacher = self.create_self_paced_teacher(with_callback=False)
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, episodes_per_update=self.EP_PER_UPDATE,
                                   context_visible=False, use_undiscounted_reward=False,
                                   context_post_processing=UnlockPickupContext)
        elif self.curriculum.acl():
            teacher = ACL(self.ALL_CONTEXTS.shape[0], self.ACL_ETA, eps=self.ACL_EPS)
            env = ACLWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False,
                             context_post_processing=lambda x: UnlockPickupContext(np.copy(self.ALL_CONTEXTS[x])))
        elif self.curriculum.plr():
            teacher = PLR(0, self.ALL_CONTEXTS.shape[0], self.PLR_REPLAY_RATE, self.PLR_BUFFER_SIZE, self.PLR_BETA,
                          self.PLR_RHO, is_discrete=True)
            env = PLRWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False,
                             context_post_processing=lambda x: UnlockPickupContext(np.copy(self.ALL_CONTEXTS[x])))
        elif self.curriculum.vds():
            teacher = VDS(0, self.ALL_CONTEXTS.shape[0], self.DISCOUNT_FACTOR, self.VDS_NQ, is_discrete=True,
                          net_arch={"feature_extractor": VDSFeatureExtractor(64, 7, 10),
                                    "layers": [64, 64], "act_func": torch.nn.ReLU()},
                          q_train_config={"replay_size": 5 * self.STEPS_PER_ITER, "lr": self.VDS_LR,
                                          "n_epochs": self.VDS_EPOCHS, "batches_per_epoch": self.VDS_BATCHES,
                                          "steps_per_update": 2 * self.STEPS_PER_ITER})
            env = VDSWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False,
                             context_post_processing=lambda x: UnlockPickupContext(np.copy(self.ALL_CONTEXTS[x])))
        elif self.curriculum.random():
            teacher = UniformDiscreteSampler(self.ALL_CONTEXTS)
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=False,
                              context_post_processing=UnlockPickupContext)
        else:
            raise RuntimeError("Invalid learning type")

        return env, DummyVecEnv([lambda: env])

    def create_learner_params(self):
        return dict(common=dict(gamma=self.DISCOUNT_FACTOR, seed=self.seed, verbose=0, device="cpu",
                                policy_kwargs=dict(features_extractor_class=UnlockPickupFeatureExtractor,
                                                   net_arch=[64, 64], activation_fn=torch.nn.ReLU)),
                    dqn=dict(buffer_size=1000000, batch_size=256, train_freq=4, exploration_final_eps=0.1, tau=0.005,
                             target_update_interval=1, learning_rate=1e-4))

    def create_experiment(self):
        timesteps = 2500000

        env, vec_env = self.create_environment(evaluation=False)
        model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

        if isinstance(env, PLRWrapper):
            env.learner = interface

        if isinstance(env, VDSWrapper):
            state_provider = lambda contexts: self.INITIAL_STATES[contexts]
            env.teacher.initialize_teacher(env, interface, state_provider)

        callback_params = {"learner": interface, "env_wrapper": env, "save_interval": 5,
                           "step_divider": self.STEPS_PER_ITER}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self, with_callback=False):
        if self.curriculum.self_paced():
            raise RuntimeError("Self-Paced Learning not supported in this environment")
        else:
            return DiscreteCurrOT(self.INIT_SAMPLES.repeat(10, axis=0), self.target_sampler,
                                  self.DELTA, self.METRIC_EPS, self.neighbour_oracle, self.distance_function,
                                  wait_until_threshold=True)

    def get_env_name(self):
        return f"unlock_pickup"

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
