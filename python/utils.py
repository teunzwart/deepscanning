import copy
import sys

import numpy as np


def change_state(state, action):
    new_state = copy.copy(state)
    new_state[action[0]] -= 1
    new_state[action[1]] += 1
    return new_state


def map_to_entire_space(bethe_numbers, max_I):
    """Map Bethe numbers to a vector of 1s and 0s on the interval [-max_I, max_I]."""
    # TODO: Fix this.
    if len(bethe_numbers) % 2 == 0:
        print(bethe_numbers)
        sys.exit("Can't handle even state mapping.")
    transformed_bethe_numbers = np.array(bethe_numbers, dtype=np.int) + max_I
    entire_space = np.zeros(2 * max_I + 1)
    for i, _ in enumerate(entire_space):
        if i in transformed_bethe_numbers:
            entire_space[i] = 1
    return entire_space


def map_to_bethe_numbers(entire_space, max_I):
    """Map a vector of 1s and 0s to Bethe numbers."""
    bethe_numbers = []
    for i, k in enumerate(entire_space):
        if k == 1:
            bethe_numbers.append(i)
    # print(bethe_numbers)
    return np.array(bethe_numbers) - max_I


def get_allowed_indices(state, interval_size):
    allowed = np.ones((interval_size, interval_size))
    for i, k in enumerate(state):
        # Mask removals from unoccupied sites.
        if k == 0:
            allowed[i] = 0
        # Mask additions to occupied sites.
        if k == 1:
            allowed[:, i] = 0
    return list(zip(*np.where(allowed == 1)))


def get_valid_random_action(state, interval_size):
    """
    Return a random valid random action.

    The input state should be a map of the entire momentum subspace,
    not Bethe numbers.
    """
    allowed_indices = get_allowed_indices(state, interval_size)
    return allowed_indices[np.random.choice(len(allowed_indices))]


def is_valid_action(state, action, interval_size):
    """Find if an action is valid (does not remove or add particles where that is not possible)."""
    if action in get_allowed_indices(state, interval_size):
        return True
    else:
        return False


def get_largest_allowed_Q_value(Q, state, previously_visited_states, N_world):
    for action in Q.flatten().argsort()[::-1]:
        if is_valid_action(state, np.unravel_index(action, (N_world, N_world)), N_world):
            new_state = change_state(state, np.unravel_index(action, (N_world, N_world)))
            if list(new_state) not in previously_visited_states:
                return new_state, action, Q[0][action]
