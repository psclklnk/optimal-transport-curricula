import pickle
import numpy as np
from typing import Tuple
from pathlib import Path
from operator import mul
from functools import reduce
from minigrid.core.world_object import Key, Door, Box
from minigrid.envs.blockedunlockpickup import BlockedUnlockPickupEnv


def unflatten_pos(idxs: np.ndarray, size: Tuple[int, ...], offset: Tuple[int, ...]):
    values = np.zeros(idxs.shape + (len(size),), dtype=np.uint8)

    divisor = 1
    for i, (s, o) in enumerate(zip(size, offset)):
        values[..., i] = o + (idxs // divisor) % s
        divisor *= s

    return values


def flatten_pos(pos: np.ndarray, size: Tuple[int, ...], offset: Tuple[int, ...]):
    idxs = np.zeros(pos.shape[:-1], dtype=np.int64)

    multiplier = 1
    for i, (s, o) in enumerate(zip(size, offset)):
        idxs += multiplier * (pos[..., i] - o)
        multiplier *= s

    return idxs


class UnlockPickupContext:
    AGENT_POS = 0
    DOOR_Y_POS = 2
    BOX_POS = 3
    KEY_POS = 5
    DOOR_OPEN = 7

    # We put the door_y- and box position at then end to have the same idxs when not looking at them
    FIELDS = [AGENT_POS, KEY_POS, DOOR_OPEN, BOX_POS, DOOR_Y_POS]
    SIZES = [(9, 4), (9, 4), (2,), (4, 4), (4,)]
    OFFSETS = [(1, 1), (1, 1), (0,), (6, 1), (1,)]

    N_CONTEXTS = int(np.prod(np.concatenate(SIZES, axis=0)))
    N_STATES = int(np.prod(np.concatenate(SIZES[:3], axis=0)))

    def __init__(self, np_rep: np.ndarray):
        self.np_context = np_rep.astype(np.uint8)

    def copy(self):
        return UnlockPickupContext(self.np_context.copy())

    @staticmethod
    def get_door_contexts(door_y_pos):
        mul = np.prod(np.concatenate(UnlockPickupContext.SIZES, axis=0)[:-1])
        offset = mul * (door_y_pos - UnlockPickupContext.OFFSETS[-1][0])
        return np.arange(mul) + offset

    # These static methods allow to process flattened version of the context and support batching. The class is simply
    # a wrapper that allows for object-oriented access to a specific entry
    @staticmethod
    def is_valid(np_contexts):
        agent_pos = np_contexts[..., UnlockPickupContext.AGENT_POS:UnlockPickupContext.AGENT_POS + 2]
        door_y_pos = np_contexts[..., UnlockPickupContext.DOOR_Y_POS]
        door_open = UnlockPickupContext.is_door_open(np_contexts)
        key_pos = np_contexts[..., UnlockPickupContext.KEY_POS:UnlockPickupContext.KEY_POS + 2]
        box_pos = np_contexts[..., UnlockPickupContext.BOX_POS: UnlockPickupContext.BOX_POS + 2]

        # First check if positions are valid. We allow key to only be in the left room and the box to be only
        # in the right room. The agent can be in both rooms
        door_pos_valid = np.logical_and(1 <= door_y_pos, door_y_pos <= 4)
        box_pos_valid = np.logical_and(np.logical_and(6 <= box_pos[..., 0], box_pos[..., 0] <= 9),
                                       np.logical_and(1 <= box_pos[..., 1], box_pos[..., 1] <= 4))

        # For the key and the agent we require the same rules: The object is only in the right room if the door is open,
        # the object cannot be in the wall (except for in the door if it is open) and finally it cannot be on the same
        # position as the box
        x_upper_bound = np.where(door_open, 9, 4)
        key_pos_valid = np.logical_and(np.logical_and(1 <= key_pos[..., 0], key_pos[..., 0] <= x_upper_bound),
                                       np.logical_and(1 <= key_pos[..., 1], key_pos[..., 1] <= 4))
        key_pos_valid = np.logical_and(key_pos_valid,
                                       np.logical_or(key_pos[..., 0] != 5, key_pos[..., 1] == door_y_pos))
        key_pos_valid = np.logical_and(key_pos_valid, np.any(key_pos != box_pos, axis=-1))

        agent_pos_valid = np.logical_and(np.logical_and(1 <= agent_pos[..., 0], agent_pos[..., 0] <= x_upper_bound),
                                         np.logical_and(1 <= agent_pos[..., 1], agent_pos[..., 1] <= 4))
        agent_pos_valid = np.logical_and(agent_pos_valid,
                                         np.logical_or(agent_pos[..., 0] != 5, agent_pos[..., 1] == door_y_pos))
        agent_pos_valid = np.logical_and(agent_pos_valid, np.any(agent_pos != box_pos, axis=-1))

        # All of the above statements need to be true for the context to be valid
        return np.all(np.stack((door_pos_valid, key_pos_valid, box_pos_valid, agent_pos_valid), axis=-1), axis=-1)

    @staticmethod
    def get_agent_pos(np_contexts: np.ndarray):
        return np_contexts[..., UnlockPickupContext.AGENT_POS: UnlockPickupContext.AGENT_POS + 2].astype(
            np.int16)

    @staticmethod
    def get_door_y_pos(np_contexts: np.ndarray):
        return np_contexts[..., UnlockPickupContext.DOOR_Y_POS].astype(np.int16)

    @staticmethod
    def get_box_pos(np_contexts: np.ndarray):
        return np_contexts[..., UnlockPickupContext.BOX_POS: UnlockPickupContext.BOX_POS + 2].astype(
            np.int16)

    @staticmethod
    def get_key_pos(np_contexts: np.ndarray):
        return np_contexts[..., UnlockPickupContext.KEY_POS: UnlockPickupContext.KEY_POS + 2].astype(
            np.int16)

    @staticmethod
    def is_key_taken(np_contexts: np.ndarray):
        key_pos = np_contexts[..., UnlockPickupContext.KEY_POS: UnlockPickupContext.KEY_POS + 2]
        agent_pos = np_contexts[..., UnlockPickupContext.AGENT_POS: UnlockPickupContext.AGENT_POS + 2]

        return np.all(key_pos == agent_pos, axis=-1)

    @staticmethod
    def is_door_open(np_contexts: np.ndarray):
        return np_contexts[..., UnlockPickupContext.DOOR_OPEN] > 0

    @staticmethod
    def get_door_pos(np_contexts: np.ndarray):
        return np.stack((5 * np.ones(np_contexts.shape[:-1], dtype=np_contexts.dtype),
                         np_contexts[..., UnlockPickupContext.DOOR_Y_POS]), axis=-1).astype(np.int16)

    @staticmethod
    def is_left_room(np_contexts: np.ndarray):
        return np_contexts[..., UnlockPickupContext.AGENT_POS] < 5

    @staticmethod
    def get_index(np_contexts: np.ndarray):
        idxs = np.zeros(np_contexts.shape[:-1], dtype=np.int64)
        multiplier = 1
        for field_start, size, offset in zip(UnlockPickupContext.FIELDS, UnlockPickupContext.SIZES,
                                             UnlockPickupContext.OFFSETS):
            field_size = len(size)
            values = np_contexts[..., field_start:field_start + field_size]
            assert np.all(unflatten_pos(flatten_pos(values, size, offset), size, offset) == values)
            idxs += multiplier * flatten_pos(values, size, offset)
            multiplier *= reduce(mul, size)

        return idxs

    @staticmethod
    def get_array(idxs: np.ndarray):
        np_contexts = np.zeros(idxs.shape + (UnlockPickupContext.DOOR_OPEN + 1,), dtype=np.uint8)
        multiplier = 1
        for field_start, size, offset in zip(UnlockPickupContext.FIELDS, UnlockPickupContext.SIZES,
                                             UnlockPickupContext.OFFSETS):
            field_size = len(size)
            local_size = reduce(mul, size)
            local_idx = (idxs // multiplier) % local_size
            np_contexts[..., field_start:field_start + field_size] = unflatten_pos(local_idx, size, offset)
            assert np.all(flatten_pos(unflatten_pos(local_idx, size, offset), size, offset) == local_idx)
            multiplier *= local_size

        return np_contexts

    def __eq__(self, other):
        if isinstance(other, UnlockPickupContext):
            return UnlockPickupContext.to_tuple(self).__eq__(UnlockPickupContext.to_tuple(other))
        else:
            return False

    def __lt__(self, other):
        return UnlockPickupContext.to_tuple(self) < UnlockPickupContext.to_tuple(other)

    def __hash__(self):
        return UnlockPickupContext.to_tuple(self).__hash__()

    def __str__(self):
        return f"Agent {self.agent_pos}, Key {self.key_pos}, Block, {self.block_pos}, Box {self.box_pos}, " \
               f"Door Y-Pos {self.door_y_pos}, Door Open {self.door_open}"

    @property
    def agent_pos(self):
        return UnlockPickupContext.get_agent_pos(self.np_context)

    @property
    def door_y_pos(self):
        return UnlockPickupContext.get_door_y_pos(self.np_context)

    @property
    def box_pos(self):
        return UnlockPickupContext.get_box_pos(self.np_context)

    @property
    def key_pos(self):
        return UnlockPickupContext.get_key_pos(self.np_context)

    @property
    def key_taken(self):
        return UnlockPickupContext.is_key_taken(self.np_context)

    @property
    def door_open(self):
        return UnlockPickupContext.is_door_open(self.np_context)

    @property
    def door_pos(self):
        return UnlockPickupContext.get_door_pos(self.np_context)

    @property
    def left_room(self):
        return UnlockPickupContext.is_left_room(self.np_context)

    def to_tuple(self):
        return tuple(self.np_context)

    def to_array(self):
        return self.np_context.copy()

    def to_index(self):
        return UnlockPickupContext.get_index(self.np_context)

    @staticmethod
    def from_index(index: np.ndarray):
        return UnlockPickupContext(UnlockPickupContext.get_array(index))

    @staticmethod
    def from_env(env: BlockedUnlockPickupEnv):
        context_info = {Key: None, Box: None, Door: None}

        # We iterate over the grid
        for j in range(env.grid.height):
            for i in range(env.grid.width):
                c = env.grid.get(i, j)
                if c is not None:
                    if isinstance(c, Door):
                        assert context_info[Door] is None

                        # We already set the door to be open even if it is only unlocked
                        context_info[Door] = (j, not c.is_locked)
                    else:
                        obj_type = type(c)
                        if obj_type in context_info:
                            assert context_info[obj_type] is None
                            context_info[obj_type] = (i, j)

        agent_pos = env.agent_pos
        if env.carrying is not None:
            obj_type = type(env.carrying)
            if obj_type in context_info:
                assert context_info[obj_type] is None
                context_info[obj_type] = agent_pos

        for v in context_info.values():
            assert v is not None

        np_context = np.zeros(UnlockPickupContext.DOOR_OPEN + 1, dtype=np.uint8)
        np_context[UnlockPickupContext.AGENT_POS] = int(agent_pos[0])
        np_context[UnlockPickupContext.AGENT_POS + 1] = int(agent_pos[1])
        np_context[UnlockPickupContext.DOOR_Y_POS] = context_info[Door][0]
        np_context[UnlockPickupContext.BOX_POS] = int(context_info[Box][0])
        np_context[UnlockPickupContext.BOX_POS + 1] = int(context_info[Box][1])
        np_context[UnlockPickupContext.KEY_POS] = int(context_info[Key][0])
        np_context[UnlockPickupContext.KEY_POS + 1] = int(context_info[Key][1])
        np_context[UnlockPickupContext.DOOR_OPEN] = 1 if context_info[Door][1] else 0

        return UnlockPickupContext(np_context)

    @staticmethod
    def from_desc(agent_pos: Tuple[int, int], door_y_pos: int, key_pos: Tuple[int, int], box_pos: Tuple[int, int],
                  door_open: bool = False):
        np_context = np.zeros(UnlockPickupContext.DOOR_OPEN + 1, dtype=np.uint8)
        np_context[UnlockPickupContext.AGENT_POS] = agent_pos[0]
        np_context[UnlockPickupContext.AGENT_POS + 1] = agent_pos[1]
        np_context[UnlockPickupContext.DOOR_Y_POS] = door_y_pos
        np_context[UnlockPickupContext.BOX_POS] = box_pos[0]
        np_context[UnlockPickupContext.BOX_POS + 1] = box_pos[1]
        np_context[UnlockPickupContext.KEY_POS] = key_pos[0]
        np_context[UnlockPickupContext.KEY_POS + 1] = key_pos[1]
        np_context[UnlockPickupContext.DOOR_OPEN] = 1 if door_open else 0

        return UnlockPickupContext(np_context)


def sum_uint8_capped(*args):
    acc = args[0].astype(np.int16)
    at_max = acc >= np.iinfo(np.uint8).max
    for idx in range(1, len(args)):
        acc = acc + args[idx]
        at_max = np.logical_or(at_max, acc >= np.iinfo(np.uint8).max)
    acc[at_max] = 255
    return acc.astype(np.uint8)


def add_uint8_capped(acc, arg):
    acc_max = acc == np.iinfo(np.uint8).max
    is_max = arg == np.iinfo(np.uint8).max
    acc += arg
    acc[np.logical_or(acc_max, is_max)] = np.iinfo(np.uint8).max


def base_distance(c1: np.ndarray, c2: np.ndarray):
    door_y_pos1 = UnlockPickupContext.get_door_y_pos(c1)
    door_y_pos2 = UnlockPickupContext.get_door_y_pos(c2)
    diff_door_pos = door_y_pos1 != door_y_pos2

    door_open1 = UnlockPickupContext.is_door_open(c1)
    door_open2 = UnlockPickupContext.is_door_open(c2)
    door_state = np.minimum(door_open1, door_open2)
    agent_pos1 = UnlockPickupContext.get_agent_pos(c1)
    agent_pos2 = UnlockPickupContext.get_agent_pos(c2)

    key_pos1 = UnlockPickupContext.get_key_pos(c1)
    key_pos2 = UnlockPickupContext.get_key_pos(c2)
    key_same = np.all(key_pos1 == key_pos2, axis=-1)

    # If the door positions differ, we use the maximum distance
    door_y_pos_bc = np.broadcast_to(door_y_pos1, np.broadcast_shapes(door_y_pos1.shape, door_y_pos2.shape))
    total_dists = object_distance(agent_pos1, key_pos1, door_y_pos1, door_open1) + \
                  object_distance(key_pos1, key_pos2, door_y_pos_bc, door_state) + \
                  object_distance(key_pos2, agent_pos2, door_y_pos2, door_open2)
    # If the keys are the same, we do not need to take the detour via the key
    total_dists[key_same] = object_distance(agent_pos1, agent_pos2, door_y_pos1, door_state)[key_same]

    # Finally, we account for the difference in box positions
    box_pos1 = UnlockPickupContext.get_box_pos(c1)
    box_pos2 = UnlockPickupContext.get_box_pos(c2)
    total_dists = sum_uint8_capped(total_dists, np.sum(np.abs(box_pos1 - box_pos2), axis=-1))

    # Context with different door positions are incomparable under this distance function
    total_dists[diff_door_pos] = np.iinfo(total_dists.dtype).max

    # Check that for equal contexts the distance is 0
    assert np.all(total_dists[np.all(c1 == c2, axis=-1)] == 0)
    return total_dists


def object_distance(pos1: np.ndarray, pos2: np.ndarray, door_y_pos: np.ndarray, door_open: np.ndarray):
    left_room1 = pos1[..., 0] < 5
    left_room2 = pos2[..., 0] < 5

    # The distance will be uint8, although we need to ensure during subtraction that the datatypes are int16
    pos1 = pos1.astype(np.int16)
    pos2 = pos2.astype(np.int16)

    res = np.zeros(np.broadcast_shapes(left_room1.shape, left_room2.shape), dtype=np.uint8)
    diff_room = left_room1 != left_room2
    # We first compute the distance for the different door setting
    res[:] = np.abs(pos1[..., 0] - pos2[..., 0]) + \
             np.abs(pos1[..., 1] - door_y_pos) + np.abs(pos2[..., 1] - door_y_pos)
    # We overwrite it with the shorter distance if the agents are in the same room
    res[~diff_room] = np.sum(np.abs(pos1 - pos2), axis=-1)[~diff_room]
    # Then we mask the invalid settings
    assert not np.any(np.logical_and(diff_room, ~door_open))

    return res


def compute_door_representative(contexts: np.ndarray):
    door_open = UnlockPickupContext.is_door_open(contexts)
    door_contexts = np.zeros_like(contexts)

    door_y_pos = UnlockPickupContext.get_door_y_pos(contexts)
    door_contexts[..., UnlockPickupContext.AGENT_POS] = 4
    door_contexts[..., UnlockPickupContext.AGENT_POS + 1] = door_y_pos
    door_contexts[..., UnlockPickupContext.DOOR_Y_POS] = door_y_pos
    door_contexts[..., UnlockPickupContext.KEY_POS] = 4
    door_contexts[..., UnlockPickupContext.KEY_POS + 1] = door_y_pos
    door_contexts[..., UnlockPickupContext.BOX_POS] = contexts[..., UnlockPickupContext.BOX_POS]
    door_contexts[..., UnlockPickupContext.BOX_POS + 1] = contexts[..., UnlockPickupContext.BOX_POS + 1]
    door_contexts[..., UnlockPickupContext.DOOR_OPEN] = 1

    return np.where(door_open[..., None], contexts, door_contexts)


def context_distance(con1: np.ndarray, con2: np.ndarray):
    squeeze = False
    if len(con1.shape) == 1:
        con1 = con1[None, :]
        squeeze = True

    if len(con2.shape) == 1:
        con2 = con2[None, :]
        squeeze = True

    # We at maximum allow for 2D contexts (such that this function creates a matrix) - 3rd dimension is the actual
    # context information
    if len(con1.shape) > 3 or len(con2.shape) > 3:
        raise RuntimeError("This function allows for at most 2D arrays of contexts")

    # If the two contexts do not have both the door open or closed, we compute the representative for opening the door
    door_state_differs = UnlockPickupContext.is_door_open(con1) != UnlockPickupContext.is_door_open(con2)

    rep_con1 = compute_door_representative(con1)
    rep_con2 = compute_door_representative(con2)

    total_distance = door_state_differs.astype(np.uint8)
    add_uint8_capped(total_distance, base_distance(con1, rep_con1))
    add_uint8_capped(total_distance, base_distance(rep_con1, rep_con2))
    add_uint8_capped(total_distance, base_distance(rep_con2, con2))

    idxs1, idxs2 = np.where(~door_state_differs)
    total_distance[idxs1, idxs2] = base_distance(con1[idxs1 % con1.shape[0], idxs2 % con1.shape[1]],
                                                 con2[idxs1 % con2.shape[0], idxs2 % con2.shape[1]])
    if squeeze:
        return np.squeeze(total_distance)
    else:
        return total_distance


def get_context_distance():
    return context_distance


class NeighbourOracle:

    def __init__(self, candidates: np.ndarray, distance_function, eta, allowed_neighbours=None):
        if allowed_neighbours is None:
            # We build a map of nearest neighbours for fast lookup during online computations
            from tqdm import tqdm
            from joblib import Parallel, delayed

            candidate_idxs = UnlockPickupContext.get_index(candidates)
            batch_size = 1000
            idx_batches = [candidate_idxs[start: start + batch_size] for start in
                           range(0, candidate_idxs.shape[0], batch_size)]

            self.allowed_neighbours = {}

            res = Parallel(n_jobs=5, batch_size=1)(
                delayed(self._process_neighbours)(distance_function, eta, idxs, candidates) for idxs in
                tqdm(idx_batches))
            for r in res:
                self.allowed_neighbours.update(r)
        else:
            self.allowed_neighbours = allowed_neighbours

    @staticmethod
    def _process_neighbours(distance_function, eta, idxs, candidates):
        allowed_neighbours = {}
        for idx, candidate in zip(idxs, candidates):
            in_region = distance_function(UnlockPickupContext.get_array(idx)[None, None], candidates[None])[0] <= eta
            allowed_neighbours[idx] = candidates[in_region]

        return allowed_neighbours

    def __call__(self, context: np.ndarray):
        assert len(context.shape) == 1
        return self.allowed_neighbours[int(UnlockPickupContext.get_index(context))]

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            return NeighbourOracle(None, None, None, allowed_neighbours=pickle.load(f))

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.allowed_neighbours, f)
