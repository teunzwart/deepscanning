import numpy as np


def right_side(I, L, N):
    """Calculate RHS of f-sum rule."""
    return 4 * np.pi**2 * N / L**4 * I**2


def left_side(list_of_states, ref_energy):
    """Calculate LHS of f-sum rule."""
    sumrule_sum = 0
    for state in list_of_states:
        sumrule_sum += (state.energy - ref_energy) * np.abs(state.ff)**2
    return sumrule_sum


def compute_average_sumrule(data, ref_energy, L, N, max_I, N_world, print_all=False):
    """
    Compute the average sumrule over all momentum slices.

    data: dictionary with momenta as keys and states as values
    ref_energy: energy of the reference state
    """
    # Catch empty dictionaries.
    if not data:
        return 0
    sumrule = 0
    for i in range(-max_I, max_I + 1):
        states = data.get(i, [])
        if i != 0:
            if print_all:
                print(f"{i:3}: {left_side(states, ref_energy) / right_side(i, L, N):.20f}")
            sumrule += (left_side(states, ref_energy) / right_side(i, L, N))
    return sumrule / (N_world - 1)
