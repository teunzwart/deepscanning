import sys

import numpy as np


def array_to_list(a):
    """
    Map element of an 2D array to a list of [value, coordinate] pairs.
    The list is sorted on the values, in decreasing order.
    """
    shape = a.shape
    value_list = []
    for i, a in enumerate(a.flatten()):
        value_list.append([a, np.unravel_index(i, shape)])
    return sorted(value_list, key=lambda x: x[0])[::-1]


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


def get_valid_random_action(state, interval_size):
    """
    Return a random valid random action.

    The input state should be a map of the entire momentum subspace,
    not Bethe numbers.
    """
    allowed = np.ones((interval_size, interval_size))
    for i, k in enumerate(state):
        # Mask removals from unoccupied sites.
        if k == 0:
            for z in range(len(state)):
                allowed[i][z] = 0
        # Mask additions to occupied sites.
        if k == 1:
            for z in range(len(state)):
                allowed[z][i] = 0
    allowed_indices = list(zip(*np.where(allowed == 1)))
    return allowed_indices[np.random.choice(len(allowed_indices))]


def is_valid_action(state, action, interval_size):
    """Find if an action is valid (does not remove or add particles where that is not possible)."""
    allowed = np.ones((interval_size, interval_size))
    for i, k in enumerate(state):
        # Mask removals from unoccupied sites.
        if k == 0:
            for z in range(len(state)):
                allowed[i][z] = 0
        # Mask additions to occupied sites.
        if k == 1:
            for z in range(len(state)):
                allowed[z][i] = 0
    allowed_indices = list(zip(*np.where(allowed == 1)))
    if action in allowed_indices:
        return True
    else:
        return False
