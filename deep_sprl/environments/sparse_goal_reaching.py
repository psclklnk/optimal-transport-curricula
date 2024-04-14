import gym
import numpy as np
from gym import utils
from deep_sprl.util.viewer import Viewer


class SparseGoalReachingEnv(gym.core.Env, utils.EzPickle):

    def __init__(self, context=np.array([0., 0.]), dict_observation: bool = False):
        """
        The env has the following shape:

        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],

        """

        self._state = None
        self.context = context
        self.max_step = 0.3
        self._dict_observation = dict_observation

        self.action_space = gym.spaces.Box(np.array([-1., -1.]), np.array([1., 1.]))
        if self._dict_observation:
            self.observation_space = gym.spaces.Dict({
                "observation": gym.spaces.Box(np.array([-9., -9.]), np.array([9., 9.])),
                "achieved_goal": gym.spaces.Box(np.array([-9., -9., 0.05]), np.array([9., 9., 18])),
                "desired_goal": gym.spaces.Box(np.array([-9., -9., 0.05]), np.array([9., 9., 18]))
            })
        else:
            self.observation_space = gym.spaces.Box(np.array([-9., -9.]), np.array([9., 9.]))

        self._viewer = Viewer(20, 20, background=(255, 255, 255))

        gym.core.Env.__init__(self)
        utils.EzPickle.__init__(**locals())

    @staticmethod
    def sample_initial_state(n=None):
        if n is None:
            return np.random.uniform(-7., -5., size=(2,))
        else:
            return np.random.uniform(-7., -5., size=(n, 2))

    def reset(self):
        self._state = self.sample_initial_state()
        if self._dict_observation:
            return {"observation": np.copy(self._state), "desired_goal": self.context,
                    "achieved_goal": np.concatenate((self._state, [self.context[2]]))}
        else:
            return np.copy(self._state)

    @staticmethod
    def _is_feasible(context):
        # Check that the context is not in or beyond the outer wall
        if np.any(context < -7.) or np.any(context > 7.):
            return False
        # Check that the context is not within the inner rectangle (i.e. in [-5, 5] x [-5, 5])
        elif np.all(np.logical_and(-5. < context, context < 5.)):
            return False
        else:
            return True

    @staticmethod
    def _project_back(old_state, new_state):
        # Project back from the bounds
        new_state = np.clip(new_state, -7., 7.)

        # Project back from the inner circle
        if -5 < new_state[0] < 5 and -5 < new_state[1] < 5:
            new_state = np.where(np.logical_and(old_state <= -5, new_state > -5), -5, new_state)
            new_state = np.where(np.logical_and(old_state >= 5, new_state < 5), 5, new_state)

        return new_state

    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.where(np.linalg.norm(desired_goal[..., :2] - achieved_goal[..., :2], axis=-1) <
                        desired_goal[..., 2], 0., -1.)  # scale = 0.1

    def step(self, action):
        if self._state is None:
            raise RuntimeError("State is None! Be sure to reset the environment before using it")

        action = self.max_step * (action / max(1., np.linalg.norm(action)))
        new_state = self._project_back(self._state, self._state + action)
        ext_reward = self.compute_reward(new_state, self.context, None)

        info = {"success": (np.linalg.norm(self.context[:2] - new_state) < self.context[2])}
        info["reward"] = 1. if info["success"] else 0.
        self._state = np.copy(new_state)

        if self._dict_observation:
            return {"observation": new_state, "desired_goal": self.context,
                    "achieved_goal": np.concatenate((new_state, [self.context[2]]))}, ext_reward, info["success"], info
        else:
            return new_state, ext_reward, info["success"], info

    def render(self, mode='human'):
        offset = 10

        outer_border_poly = [np.array([-9, 1]), np.array([9, 1]), np.array([9, -1]), np.array([-9, -1])]

        self._viewer.polygon(np.array([0, -8]) + offset, 0, outer_border_poly, color=(0, 0, 0))
        self._viewer.polygon(np.array([0, 8]) + offset, 0, outer_border_poly, color=(0, 0, 0))
        self._viewer.polygon(np.array([-8, 0]) + offset, 0.5 * np.pi, outer_border_poly, color=(0, 0, 0))
        self._viewer.polygon(np.array([8, 0]) + offset, 0.5 * np.pi, outer_border_poly, color=(0, 0, 0))
        self._viewer.square(np.zeros(2) + offset, 0., 10, color=(0, 0, 0))

        self._viewer.circle(self.context[:2] + offset, 0.25, color=(255, 0, 0))
        self._viewer.circle(self._state + offset, 0.25, color=(0, 0, 0))
        self._viewer.display(0.01)


if __name__ == "__main__":
    env = SparseGoalReachingEnv()

    env.reset()
    env._state = np.array([0, -6])
    env.render()
    while True:
        env.step(np.array([-1., 1]))
        env.render()
