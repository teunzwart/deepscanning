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


def no_of_particle_hole_pairs(state, reference_state, N):
    return N - len(set(np.where(state == 1)[0]).intersection(np.where(np.array(reference_state) == 1)[0]))


def select_action(available_actions, state, previously_visited_states, max_I, N_world, N, check_no_of_pairs=True):
    """
    Select the action with the highest Q-value, subject to constraint.
    
    Contraints are: sum below max_I, not previously visited, minimal number of ph-pairs (optional)
    """
    no_of_ph_per_action = []

    for action in available_actions:
        if is_valid_action(state, action, N_world):
            new_state = change_state(state, action)
            if (list(new_state) not in previously_visited_states) and (abs(sum(map_to_bethe_numbers(new_state, max_I))) <= max_I):
                if check_no_of_pairs:
                    no_of_pairs = no_of_particle_hole_pairs(new_state, previously_visited_states[0], N)
                    if no_of_pairs == 1:
                        return new_state, action
                    else:
                        no_of_ph_per_action.append((no_of_pairs, (new_state, action)))
                        continue
                else:
                    return new_state, action


    return sorted(no_of_ph_per_action, key=lambda x: x[0])[0][1]


def measure_with_error(data, method):
    measure = f"{data.mean():.0f}"
    error = f"{data.std():.0f}"
    print(f"{method}: \\num{{{measure} \\pm {error}}}")
