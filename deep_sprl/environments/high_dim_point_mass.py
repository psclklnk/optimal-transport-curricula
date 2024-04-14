import numpy as np
from scipy.interpolate import interp1d
from gym import Env, spaces
from deep_sprl.util.viewer import Viewer


class Gate:

    def __init__(self, x_pos: float, width_left: float, width_right: float, y_start: float, y_end: float):
        self.y_start = y_start
        self.y_end = y_end

        # We create the control points
        self.left_border = x_pos - width_left
        self.right_border = x_pos + width_right

        assert self.y_start > self.y_end
        assert self.y_start > 0 > self.y_end

    def crash(self, x_start: np.ndarray, x_end: np.ndarray):
        in_gate = self.y_end <= x_start[2] <= self.y_start or self.y_end <= x_end[2] <= self.y_start
        if x_end[2] < x_start[2]:
            in_gate = in_gate or (x_start[2] > self.y_start >= x_end[2])
        else:
            in_gate = in_gate or (x_start[2] < self.y_end <= x_end[2])

        if in_gate:
            max_y_pos = max(x_start[2], x_end[2])
            min_y_pos = min(x_start[2], x_end[2])
            probe_points = np.linspace(min(max_y_pos, self.y_start), max(min_y_pos, self.y_end), 100)
            assert np.all(np.logical_and(max_y_pos >= probe_points, min_y_pos <= probe_points))

            x_int = interp1d([x_start[2], x_end[2]], [x_start[0], x_end[0]])(probe_points)
            crash = np.logical_or(x_int < self.left_border, x_int > self.right_border)
            if np.any(crash):
                crash_state = np.zeros(4)
                crash_idx = np.argmax(crash)
                crash_state[0] = x_int[crash_idx]
                crash_state[2] = probe_points[crash_idx]
                return True, crash_state
            else:
                return False, x_end
        else:
            return False, x_end


class HighDimPointMass(Env):

    def __init__(self, dim, max_reduction=True):
        self._state = None
        self._goal_state = np.array([0., 0., -3., 0.])
        self.context = np.array([0.] * dim + [2.] * 2 * dim)
        self.max_reduction = max_reduction

        self.action_space = spaces.Box(np.array([-10., -10.]), np.array([10., 10.]))
        self.observation_space = spaces.Box(np.array([-4., -np.inf, -4., -np.inf, np.inf, np.inf, np.inf]),
                                            np.array([4., np.inf, 4., np.inf, -np.inf, -np.inf, -np.inf]))

        self.gate = None
        self.wall_y_start = 5e-1
        self.wall_y_end = -5e-1
        self._dt = 0.01
        self._viewer = Viewer(8, 8, background=(255, 255, 255))

    def reduced_context(self):
        context_dim = self.context.shape[0] // 3
        max_abs_pos = np.argmax(np.abs(self.context[:context_dim]))
        if self.max_reduction:
            return np.array([self.context[max_abs_pos],
                             np.max(self.context[context_dim:2 * context_dim]),
                             np.max(self.context[2 * context_dim:])])
        else:
            return np.array([self.context[max_abs_pos],
                             np.min(self.context[context_dim:2 * context_dim]),
                             np.min(self.context[2 * context_dim:])])

    def reset(self):
        reduced_context = self.reduced_context()
        self.gate = Gate(reduced_context[[0]], reduced_context[[1]], reduced_context[[2]],
                         self.wall_y_start, self.wall_y_end)
        self._state = np.array([0., 0., 3., 0.])
        return np.concatenate((self._state / 2., self.reduced_context()))

    def _step_internal(self, state, action):
        state_der = np.zeros(4)
        state_der[0::2] = state[1::2]
        state_der[1::2] = 1.5 * action - 0. * state[1::2] + np.random.normal(0, 0.05, (2,))
        new_state = np.clip(state + self._dt * state_der, self.observation_space.low[:4],
                            self.observation_space.high[:4])

        return self.gate.crash(state, new_state)

    def step(self, action):
        if self._state is None:
            raise RuntimeError("State is None! Be sure to reset the environment before using it")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        new_state = self._state
        crash = False
        for i in range(0, 10):
            crash, new_state = self._step_internal(new_state, action)
            if crash:
                break

        self._state = np.copy(new_state)

        info = {"success": np.linalg.norm(self._goal_state[0::2] - new_state[0::2]) < 0.25}

        return np.concatenate((new_state / 2., self.reduced_context())), \
            np.exp(-0.6 * np.linalg.norm(self._goal_state[0::2] - new_state[0::2])), crash, info

    def render(self, mode='human'):
        context = self.reduced_context()
        pos = context[0] + 4.
        left_width = self.context[1]
        right_width = self.context[2]
        self._viewer.line(np.array([0., 4.]), np.array([np.clip(pos - left_width, 0., 8.), 4.]), color=(0, 0, 0),
                          width=0.2)
        self._viewer.line(np.array([np.clip(pos + right_width, 0., 8, ), 4.]), np.array([8., 4.]), color=(0, 0, 0),
                          width=0.2)

        self._viewer.line(np.array([3.9, 0.9]), np.array([4.1, 1.1]), color=(255, 0, 0), width=0.1)
        self._viewer.line(np.array([4.1, 0.9]), np.array([3.9, 1.1]), color=(255, 0, 0), width=0.1)

        self._viewer.circle(self._state[0::2] + np.array([4., 4.]), 0.1, color=(0, 0, 0))
        self._viewer.display(self._dt)
