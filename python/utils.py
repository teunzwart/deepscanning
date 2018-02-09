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
