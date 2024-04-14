import gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional
from minigrid.core.world_object import Key, Door, Box
from minigrid.envs.unlockpickup import UnlockPickupEnv
from deep_sprl.environments.unlockpickup.unlock_pickup_context import UnlockPickupContext


class ContextualUnlockPickupEnv(UnlockPickupEnv):

    def __init__(self, context_sampler: Optional[Callable[[], UnlockPickupContext]] = None, history_size: int = 1,
                 **kwargs):
        super(ContextualUnlockPickupEnv, self).__init__(**kwargs)
        self.context = None
        self.buffer = []
        self.history_size = history_size
        self.context_sampler = context_sampler
        self.post_reset_assignment = None
        self.observation_space = gym.spaces.Box(low=-np.ones((self.history_size, 147)),
                                                high=np.ones((self.history_size, 147)))
        self.action_space = gym.spaces.Discrete(7)

    def accumulate_obs(self, image_obs):
        self.buffer.append(image_obs)
        del self.buffer[0]
        return np.stack(self.buffer, axis=0)

    def reset(self, *, seed=None, options=None):
        if self.context_sampler is not None:
            self.context = self.context_sampler()

        super(UnlockPickupEnv, self).reset(seed=seed, options=options)

        if self.post_reset_assignment is not None:
            self.carrying = self.post_reset_assignment
            self.post_reset_assignment = None

        self.buffer = [np.zeros(147) for _ in range(self.history_size)]

        return self.accumulate_obs(self.gen_obs()["image"].reshape(-1))

    def step(self, action):
        obs, reward, terminated, truncated, info = super(ContextualUnlockPickupEnv, self).step(action)
        info["TimeLimit.truncated"] = truncated

        # Minigrid internally decays the rewards with increasing number of steps. This does not make sense since
        # otherwise the time-step would need to be part of the state (which we do not do here since it really is
        # unnecessary)
        reward = 1. if reward > 0 else 0.

        return self.accumulate_obs(obs["image"].reshape(-1)), reward, terminated or truncated, info

    def render(self, **kwargs):
        return super(ContextualUnlockPickupEnv, self).render()

    def _gen_grid(self, width, height):
        if self.context is None:
            super()._gen_grid(width, height)
        else:
            super(UnlockPickupEnv, self)._gen_grid(width, height)

            # Add a box to the room on the right
            right_room = self.get_room(1, 0)
            box = Box(self._rand_color())
            self.grid.set(self.context.box_pos[0], self.context.box_pos[1], box)
            box.init_pos = tuple(self.context.box_pos)
            box.cur_pos = tuple(self.context.box_pos)
            right_room.objs.append(box)

            # Set the door position
            left_room = self.get_room(0, 0)
            x_m = left_room.top[0] + left_room.size[0] - 1

            if self.context.door_open:
                left_room.locked = False
                door = Door(self._rand_color(), is_locked=False, is_open=True)
            else:
                left_room.locked = True
                door = Door(self._rand_color(), is_locked=True, is_open=False)

            left_room.door_pos[0] = (x_m, self.context.door_y_pos)
            self.grid.set(x_m, self.context.door_y_pos, door)
            door.cur_pos = (x_m, self.context.door_y_pos)

            neighbor = left_room.neighbors[0]
            left_room.doors[0] = door
            neighbor.doors[(0 + 2) % 4] = door

            # Set the key to a specific position
            key = Key(door.color)
            key.init_pos = tuple(self.context.key_pos)
            if self.context.key_taken:
                self.post_reset_assignment = key
                key.cur_pos = self.context.agent_pos
            else:
                self.grid.set(*tuple(self.context.key_pos), key)
                key.cur_pos = tuple(self.context.key_pos)
                left_room.objs.append(key)

            self.agent_pos = tuple(self.context.agent_pos)
            self.agent_dir = self._rand_int(0, 4)

            self.obj = box
            self.mission = f"pick up the {box.color} {box.type}"


def visualize_contexts(contexts: List[UnlockPickupContext]):
    env = ContextualUnlockPickupEnv(context_sampler=None, render_mode="rgb_array")
    n_cols = int(np.sqrt(len(contexts)))
    n_rows = int(np.ceil(len(contexts) / n_cols))

    f, axs = plt.subplots(n_rows, n_cols, figsize=(8, 4))
    axs = np.reshape(axs, -1)
    for ax, c in zip(axs, contexts):
        env.context_sampler = lambda: c
        env.reset()
        ax.imshow(env.render())
        ax.axis("off")
    plt.tight_layout()
    plt.show()
