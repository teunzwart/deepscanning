import numpy as np
import matplotlib.pyplot as plt

import pickle

import sys

import lieb_liniger_state as lls

import psi_form_factor as pff

L = 3
N = 3
I_max = 5

def all_unique(state):
    """Check if all entries of a state are unique."""
    if len(state) > len(set(state)):
        return False
    else:
        return True


def generate_all_legal_new_states(current_state, previous_states, max_I):
    """
    Generate all new legal states for the Lieb-Liniger model.

    Legal states are all those states derived from the current state
    that mutate one of the Bethe numbers by +/- 1, have only unique
    entries, and have not previously been visited.
    """
    legal_new_states = []
    for i, bethe_number in enumerate(current_state):
        for p in [-1, 1]:
            new_state = sorted(list(current_state[:i]) + list(current_state[i+1:]) + list([current_state[i] + p]))
            if all_unique(new_state) and new_state not in previous_states and not (np.abs(np.array(new_state)) > max_I).any():
                legal_new_states.append(new_state)

    return legal_new_states


def descent_tree(root_state, max_no_of_descents=10):
    already_explored = [root_state]
    deltas = []
    edges = [root_state]
    unexpanded = [root_state]
    no_edges = []
    for k in range(1000):
        # if k % 100 == 0:
        print(k)
        no_edges.append(len(edges))

        # to_explore = edges[np.random.randint(0, len(edges))]
        to_explore, delta = find_lowest_energy_delta(already_explored[-1], unexpanded)
        deltas.append(delta)
        already_explored.append(to_explore)
        edges.remove(to_explore)
        unexpanded.remove(to_explore)
        descendants = generate_all_legal_new_states(to_explore, already_explored)
        for k in descendants:
            edges.append(k)
            unexpanded.append(k)

    print("plotting")
    plt.plot(range(1000), no_edges)
    plt.show()

    
def find_lowest_energy_delta(base, edges):
    # print("start")
    # print(base)
    energy_deltas = []
    N = len(edges[0])
    base_state = lls.lieb_liniger_state(1, L, N, base)
    base_state.calculate_all()
    base_energy  = base_state.energy
    # print("base", base_energy)
    lowest_energy_diff = 10000000000
    index = -1
    for i, k in enumerate(edges):
        state = lls.lieb_liniger_state(1, L, N, k)
        state.calculate_all()
        state_energy = state.energy
        # print("state", state_energy)
        diff = state_energy - base_energy
        # print("diff", diff)
        if diff < lowest_energy_diff:
            # print(k)
            lowest_energy_diff = diff
            # print(lowest_energy_diff)
            index = i
    return edges[index], lowest_energy_diff


if __name__ == "__main__":
    # state = lls.lieb_liniger_state(1, L, 10)
    # descent_tree(list(state.Is), 1)
    added_data = []
    removed_data = []

    for k in range(10000):
        if k % 100 == 0:
            print(k)
        rstate = lls.lieb_liniger_state(1, N, N, lls.generate_bethe_numbers(N, [], I_max))
        orig_bethe_map = map_to_entire_space(rstate.Is, I_max)
        # print(orig_bethe_map)
        rstate.calculate_all()
        # print(rstate.Is)
        adjacent_states = generate_all_legal_new_states(rstate.Is, [], I_max)
        # print(adjacent_states)

        ffs = {}
        for k in adjacent_states:
            lstate = lls.lieb_liniger_state(1, N, N, k)
            lstate.calculate_all()
            ff = pff.calculate_normalized_form_factor(lstate, rstate)
            ffs[np.abs(ff*ff)] = lstate.Is
        # print(ffs)
        # print(ffs[max(ffs)])
        map_highest_ff = map_to_entire_space(ffs[max(ffs)], I_max)
        # print(map_highest_ff)
        delta = map_highest_ff - orig_bethe_map
        # print(delta)

        removed_index = np.where(delta == -1)
        added_index = np.where(delta == 1)
        removed = np.zeros(2 * I_max + 1)
        added = np.zeros(2 * I_max + 1)

        np.put(removed, removed_index, 1)
        np.put(added, added_index, 1)
        # print("rem", removed)
        # print("add", added)

        added_data.append({"rstate": orig_bethe_map, "delta": added})
        removed_data.append({"rstate": orig_bethe_map, "delta": removed})

    with open('removed.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(removed_data, f, pickle.HIGHEST_PROTOCOL)

    with open('added.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(added_data, f, pickle.HIGHEST_PROTOCOL)
