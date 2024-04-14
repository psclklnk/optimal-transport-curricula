import numpy as np
from gym import Env, spaces
import matplotlib.pyplot as plt
from typing import Optional, List
from scipy.sparse import csr_matrix
from deep_sprl.util.viewer import Viewer
from scipy.sparse.csgraph import dijkstra


class ContextualFiniteMDP(Env):
    """
    Contextual Finite Markov Decision Process with a discrete context space and a shared state-action space. Dynamics
    and rewards can change arbitrarily
    """

    def __init__(self, ps: List[np.ndarray], nss: List[np.ndarray], rews: List[np.ndarray], abss: List[np.ndarray],
                 context_representatives: np.ndarray, state_representatives: np.ndarray,
                 mus: Optional[List[np.ndarray]] = None, background_visualization: Optional[List[np.ndarray]] = None):
        """
        Constructor.
        Args:
            ps (np.ndarray): list of transition probability matrices of shape n_s x n_a x n_ns
            nss (np.ndarray): list of next_state_matrices of shape n_s x n_a x n_ns where n_ns is the number of
                             follow up states
            rews (np.ndarray): list of reward matrices of shape n_s x n_a x n_ns;
            abss (np.ndarray): list of absorbing flags for the states;
            state_representatives (np.ndarray): A map from the discrete states to continuous representations
            context_representatives (np.ndarray): A map from the discrete contexts to continuous representations
            mus (np.ndarray, None): list of initial state probability distributions. If not given, the initial state
                                   distribution will be uniform over all states
            background_visualization (np.ndarray, None): An optional background for visualizing contexts. Note that
                                                         for visualization, we assume that the states correspond to
                                                         positions on a 2D grid. Hence, the background is assumed to be
                                                         an image
        """

        n_s, n_a, n_ns = ps[0].shape

        self.action_space = spaces.Discrete(n=n_a)
        self.observation_space = spaces.Box(np.concatenate((np.min(context_representatives, axis=0),
                                                            np.min(state_representatives, axis=0))),
                                            np.concatenate((np.max(context_representatives, axis=0),
                                                            np.max(state_representatives, axis=0))))

        assert state_representatives.shape[0] == n_s
        assert context_representatives.shape[0] == n_s
        self.state_representatives = state_representatives
        self.context_representatives = context_representatives

        if mus is None:
            mus = [np.ones(n_s) / n_s] * len(ps)

        assert len(ps) == len(nss)
        assert len(ps) == len(rews)
        assert len(ps) == len(mus)

        for p, ns, rew, mu in zip(ps, nss, rews, mus):
            assert p.shape == (n_s, n_a, n_ns)
            assert ns.shape == (n_s, n_a, n_ns)
            assert rew.shape == (n_s, n_a)
            assert mu.shape == (n_s,)

        # MDP parameters
        self.ps = np.array(ps)
        self.nss = np.array(nss)
        self.rews = np.array(rews)
        self.abss = np.array(abss)
        self.mus = np.array(mus)
        self.n_a = n_a

        # The context needs to be set from the outside
        self.context = None

        self._state = None
        self._viewer = Viewer(8, 8, background=(255, 255, 255))
        self._dt = 1 / 60
        self.background_visualization = background_visualization

        if self.background_visualization is not None:
            cm = plt.get_cmap('cividis')
            self.background_visualization = [np.floor(255 * cm(bv)).astype(np.uint8)[..., :-1] for bv in
                                             self.background_visualization]

        super().__init__()

    def _uniform_context_sampler(self):
        return np.random.randint(0, len(self.ps))

    def get_continuous_state(self):
        return np.concatenate((self.context_representatives[self.context], self.state_representatives[self._state]))

    def reset(self, state=None):
        if state is None:
            self._state = np.random.choice(self.mus[self.context].size, p=self.mus[self.context])
        else:
            self._state = state

        return self.get_continuous_state()

    def step(self, action):
        p = self.ps[self.context][self._state, action, :]
        next_state_idx = np.random.choice(p.size, p=p)
        next_state = self.nss[self.context][self._state, action, next_state_idx]
        absorbing = self.abss[self.context][self._state]
        reward = self.rews[self.context][self._state, action]

        self._state = next_state

        return self.get_continuous_state(), reward, absorbing, {}

    def render(self, mode="human"):
        cont_context = self.get_continuous_state()[:2]
        cont_state = self.get_continuous_state()[2:]

        if self.background_visualization is not None:
            self._viewer.background_image(self.background_visualization[self.context])

        self._viewer.circle(np.array([4., 4.]) + np.array([4, -4]) * cont_context, 0.1, color=(255, 0, 0))
        self._viewer.circle(np.array([4., 4.]) + np.array([4, -4]) * cont_state, 0.1, color=(0, 255, 0))
        self._viewer.display(self._dt)

    def get_agent_policy(self, agent):
        cidx, sidx = np.meshgrid(np.arange(self.context_representatives.shape[0]),
                                 np.arange(self.context_representatives.shape[0]), indexing="ij")

        continuous_state = np.concatenate((self.context_representatives[cidx], self.state_representatives[sidx]),
                                          axis=-1)

        th_obs = agent.policy.obs_to_tensor(np.reshape(continuous_state, (-1, 4)))[0]
        return np.reshape(agent.policy.get_distribution(th_obs).distribution.probs.detach().cpu().numpy(),
                          cidx.shape + (self.n_a,))

    def evaluate_agent(self, agent, gamma, eps=1e-4):
        if agent is None:
            return None
        else:
            policies = self.get_agent_policy(agent)

            policy_vs = []
            for pi, r, ps, ns, absorb in zip(policies, self.rews, self.ps, self.nss, self.abss):
                policy_vs.append(policy_iteration(pi, r, ps, ns, absorb, gamma, eps=eps)[0])

            policy_returns = []
            for oq, mu in zip(policy_vs, self.mus):
                policy_returns.append(np.sum(oq * mu))
            return np.array(policy_returns)


def generate_e_maze(size, u_height, noise_level=0.):
    xy = np.stack(np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size)), axis=-1)
    heights = np.zeros(xy.shape[:2])
    heights[0:int(0.8 * size), int(0.2 * size):int(0.4 * size)] = u_height
    heights[0:int(0.8 * size), int(0.6 * size):int(0.8 * size)] = u_height
    heights += np.random.uniform(-noise_level, noise_level, heights.shape)

    return xy, heights


def generate_mdp(height_map: np.ndarray, desired_state: np.ndarray, height_scaling: float = 1.,
                 height_offset: float = 5., transition_noise: float = 0.1):
    """
    Generate information for a discrete MDP that represents 2D discretized plane with differing heights in which a
    desired position is to be reached.

    There are 4 actions: Left, Right, Top, Down

    The transition probabilities are determined by a sigmoid function that depends on the height difference, i.e.
                (1 - transition_noise) * 1 / (1 + exp(height_scaling * (height_diff - height_offset)))
    Consequently, at a height difference of height_offset, the chance of successfully moving into the desired direction
    is 0.5.

    Args:
        height_map: The height map as a 2D array
        desired_state: The desired state to be reached (as a 2D index into the height map)
        height_scaling: The scaling of the sigmoid function that relates height difference and transition success
                        probability
        height_offset: The offset of the sigmoid function that relates height difference and transition success
                       probability
        transition_noise: The transition noise

    Returns: A list of states (N x 2), actions (4 x 2), rewards (N), transition probabilities (N x 4 x 2),
             follow-up states (N x 4 x 2) and absorbing states (N) to be used for constructing a ContextualFiniteMDP

    """

    height, width = height_map.shape
    states = np.stack(np.unravel_index(np.arange(width * height), (width, height)), axis=-1)
    # We add one terminal state that is transitioned to after reaching the desired state (as we have no no-op action)!
    n_states = states.shape[0]

    # We consider 4 actions for which each one can succeed or fail, depending on the terrain height
    p = np.zeros((n_states, 4, 2))
    sf = np.zeros((n_states, 4, 2), dtype=np.int64)
    absorbing = np.zeros(n_states, dtype=bool)

    actions = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]
    for action_idx in range(0, 4):
        # We do not consider the last state for now as this one is special
        new_states = states + actions[action_idx]
        new_states[:, 0] = np.clip(new_states[:, 0], 0, height_map.shape[0] - 1)
        new_states[:, 1] = np.clip(new_states[:, 1], 0, height_map.shape[1] - 1)

        height_diff = np.abs(height_map[new_states[:, 0], new_states[:, 1]] -
                             height_map[states[:, 0], states[:, 1]])

        success_prob = (1 - transition_noise) * (1 / (1 + np.exp(height_scaling * (height_diff - height_offset))))

        sf[:, action_idx, 0] = np.ravel_multi_index([states[:, 0], states[:, 1]], height_map.shape)
        sf[:, action_idx, 1] = np.ravel_multi_index([new_states[:, 0], new_states[:, 1]], height_map.shape)
        p[:, action_idx, 0] = 1 - success_prob
        p[:, action_idx, 1] = success_prob

    # Now we change the transition dynamics of the target state and the terminal state
    flat_des_state = np.ravel_multi_index(desired_state, height_map.shape)
    p[flat_des_state, :, 0] = 0.
    p[flat_des_state, :, 1] = 1.
    sf[flat_des_state, :, 0] = flat_des_state
    sf[flat_des_state, :, 1] = flat_des_state

    absorbing[flat_des_state] = True

    r = -1 * np.repeat(np.any(states != desired_state[None, :], axis=-1).astype(np.float64)[:, None], 4, axis=1)
    r += 1
    return states, np.array(actions), r, p, sf, absorbing


class EMazeEnv(ContextualFiniteMDP):

    def __init__(self, size: int = 20, dist_fn=None):
        xy, height_map = generate_e_maze(size, u_height=200, noise_level=0)

        rewards = np.zeros((size * size, size * size, 4))
        transition_probabilities = np.zeros((size * size, size * size, 4, 2))
        next_states = np.zeros((size * size, size * size, 4, 2), dtype=np.int64)
        absorbings = np.zeros((size * size, size * size))
        for i in range(height_map.shape[0]):
            for j in range(height_map.shape[1]):
                s, a, r, p, sf, absorbing = generate_mdp(height_map, np.array([i, j]), height_scaling=1,
                                                         height_offset=5, transition_noise=0.1)
                rewards[np.ravel_multi_index((i, j), (size, size))] = r
                transition_probabilities[np.ravel_multi_index((i, j), (size, size))] = p
                next_states[np.ravel_multi_index((i, j), (size, size))] = sf
                absorbings[np.ravel_multi_index((i, j), (size, size))] = absorbing

        # We do not use xy here we shift the values to account for visualization
        value_map = np.reshape(np.stack(np.meshgrid(np.linspace(-1, 1, 21)[:-1],
                                                    np.linspace(-1, 1, 21)[:-1]), axis=-1), (-1, 2))
        value_map += 0.5 * (value_map[1, 0] - value_map[0, 0])

        # The agent always starts in the top left corner of the Maze
        mu0 = np.zeros(400)
        mu0[0] = 1.

        if dist_fn is None:
            self.dist_fn = None
        elif dist_fn == "shortest_path":
            self.dist_fn = ShortestPathDistance(xy, height_map)
        elif dist_fn == "euclidean":
            self.dist_fn = EuclideanDistance(xy, height_map)
        elif dist_fn == "opt_perf":
            self.dist_fn = OptimalPerformanceDistance(rewards, transition_probabilities, next_states, absorbings,
                                                      0.99, [np.copy(mu0) for _ in range(size * size)])
        elif dist_fn == "cur_perf":
            self.dist_fn = CurrentPerformanceDistance(self, 0.99)
        else:
            raise RuntimeError(f"Unknown distance type {dist_fn}")

        super().__init__(list(transition_probabilities), list(next_states), list(rewards), list(absorbings), value_map,
                         value_map, mus=[np.copy(mu0) for _ in range(size * size)],
                         # We need to transpose the height map for it to show properly
                         background_visualization=[height_map.T for _ in range(size * size)])


class ShortestPathDistance:

    def __init__(self, grid_positions: np.ndarray, height_map: np.ndarray):
        dists = []
        row_ids = []
        col_ids = []
        # Is a M x N x 3 array
        pos = np.concatenate((grid_positions, height_map[..., None]), axis=-1)
        for i in range(pos.shape[0]):
            for j in range(pos.shape[1]):
                if i - 1 >= 0:
                    dists.append(np.sum(np.abs(pos[i, j, :] - pos[i - 1, j, :])))
                    row_ids.append(np.ravel_multi_index((i, j), pos.shape[:2]))
                    col_ids.append(np.ravel_multi_index((i - 1, j), pos.shape[:2]))

                if i + 1 < pos.shape[0]:
                    dists.append(np.sum(np.abs(pos[i, j, :] - pos[i + 1, j, :])))
                    row_ids.append(np.ravel_multi_index((i, j), pos.shape[:2]))
                    col_ids.append(np.ravel_multi_index((i + 1, j), pos.shape[:2]))

                if j - 1 >= 0:
                    dists.append(np.sum(np.abs(pos[i, j, :] - pos[i, j - 1, :])))
                    row_ids.append(np.ravel_multi_index((i, j), pos.shape[:2]))
                    col_ids.append(np.ravel_multi_index((i, j - 1), pos.shape[:2]))

                if j + 1 < pos.shape[1]:
                    dists.append(np.sum(np.abs(pos[i, j, :] - pos[i, j + 1, :])))
                    row_ids.append(np.ravel_multi_index((i, j), pos.shape[:2]))
                    col_ids.append(np.ravel_multi_index((i, j + 1), pos.shape[:2]))

        size = np.prod(pos.shape[:2])
        self.dist_matrix, self.predecessors = dijkstra(csgraph=csr_matrix((dists, (row_ids, col_ids)),
                                                                          shape=(size, size)),
                                                       directed=False, return_predecessors=True)

    def __call__(self):
        return self.dist_matrix


class EuclideanDistance:

    def __init__(self, grid_positions: np.ndarray, height_map: np.ndarray):
        pos = np.concatenate((grid_positions, height_map[..., None]), axis=-1)
        pos = np.reshape(pos, (-1, 3))
        self.dist_matrix = np.sqrt(np.sum(np.square(pos[None, ...] - pos[:, None, :]), axis=-1))

    def __call__(self):
        return self.dist_matrix


def value_iteration(r: np.ndarray, p: np.ndarray, sf: np.ndarray, terminal: np.ndarray, gamma: float, eps: float = 1e-5,
                    save_dir=None):
    """
    Performs value iteration on the discretized MDP until convergence within eps precision

    :param r: The reward function as an array of shape (ns, na)
    :param p: The transition probabilities as an array of shape (ns, na, nns)
    :param sf: The next states corresponding to the previous transition probabilities as an array of shape (ns, na, nns)
    :param terminal: Whether this state is terminal or not
    :param gamma: The discount factor
    :param eps: The maximum allowed change in value function before termination
    :param save_dir: A Path in which to store the individual value function iterates
    :return: The optimal value function with tolerance epsilon
    """

    err = np.inf
    vf = np.zeros(r.shape[0])

    n_iter = 0
    while err > eps:
        vf_new = np.max(r + np.where(terminal[:, None], 0., gamma * np.sum(p * vf[sf], axis=-1)), axis=-1)

        if save_dir is not None:
            np.save(save_dir / ("vf_%06d.pkl" % n_iter), vf_new)

        err = np.max(np.abs(vf_new - vf))
        vf = vf_new
        n_iter += 1

    return vf, n_iter


def policy_iteration(pi: np.ndarray, r: np.ndarray, p: np.ndarray, sf: np.ndarray, terminal: np.ndarray, gamma: float,
                     eps: float = 1e-5, save_dir=None):
    """
    Performs value iteration on the discretized MDP until convergence within eps precision

    :param pi: The policy as an array of shape (ns, na)
    :param r: The reward function as an array of shape (ns, na)
    :param p: The transition probabilities as an array of shape (ns, na, nns)
    :param sf: The next states corresponding to the previous transition probabilities as an array of shape (ns, na, nns)
    :param terminal: Whether this state is terminal or not
    :param gamma: The discount factor
    :param eps: The maximum allowed change in value function before termination
    :param save_dir: A Path in which to store the individual value function iterates
    :return: The optimal value function with tolerance epsilon
    """

    err = np.inf
    vf = np.zeros(r.shape[0])

    n_iter = 0
    while err > eps:
        vf_new = np.sum(pi * (r + np.where(terminal[:, None], 0., gamma * np.sum(p * vf[sf], axis=-1))), axis=-1)

        if save_dir is not None:
            np.save(save_dir / ("vf_%06d.pkl" % n_iter), vf_new)

        err = np.max(np.abs(vf_new - vf))
        vf = vf_new
        n_iter += 1

    return vf, n_iter


class OptimalPerformanceDistance:

    def __init__(self, rewards, transition_probabilities, next_states, absorbings, discount_factor, mus, eps=1e-4):
        optimal_vs = []
        for r, ps, ns, absorb in zip(rewards, transition_probabilities, next_states, absorbings):
            optimal_vs.append(value_iteration(r, ps, ns, absorb, discount_factor, eps=eps)[0])

        optimal_returns = []
        for oq, mu in zip(optimal_vs, mus):
            optimal_returns.append(np.sum(oq * mu))
        optimal_returns = np.array(optimal_returns)

        self.distance_matrix = np.abs(optimal_returns[:, None] - optimal_returns[None, :])

    def __call__(self, *args, **kwargs):
        return self.distance_matrix


class CurrentPerformanceDistance:

    def __init__(self, mdp, discount_factor):
        self.mdp = mdp
        self.discount_factor = discount_factor
        self.agent = None

    def __call__(self, *args, **kwargs):
        if self.agent is None:
            return None
        else:
            policy_returns = self.mdp.evaluate_agent(self.agent, self.discount_factor)
            return np.abs(policy_returns[:, None] - policy_returns[None, :])


if __name__ == "__main__":
    import time

    env = EMazeEnv()
    env.context = 19

    actions = [2] * 50 + [0] * 50 + [3] * 50
    count = 0
    env.reset()
    while True:
        state, reward, absorbing, info = env.step(actions[min(count, len(actions) - 1)])
        count += 1
        env.render()

        if absorbing:
            print("Done")
            break

        time.sleep(0.1)
