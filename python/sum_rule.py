import numpy as np


def right_side(k, L, N):
    """Calculate RHS of f-sum rule."""
    return 4 * np.pi**2 * N / L**4 * k**2


def left_side(list_of_states, ref_energy):
    """Calculate LHS of f-sum rule."""
    sumrule_sum = 0
    for state in list_of_states:
        sumrule_sum += (state.energy - ref_energy) * np.abs(state.ff)**2
    return sumrule_sum


def compute_average_sumrule(data, ref_energy, L, N, print_all=False):
    """
    Compute the average sumrule over all momentum slices.

    data: dictionary with momenta as keys and states as values
    ref_energy: energy of the reference state
    """
    sumrule = 0
    no_of_momentum_slices = 0
    for momentum, states in sorted(data.items()):
        if momentum != 0:
            if print_all:
                print(f"{momentum:3}: {left_side(states, ref_energy):.20f} {right_side(momentum, L, N):.20f}  {left_side(states, ref_energy) / right_side(momentum, L, N):.20f}")
            sumrule += left_side(states, ref_energy) / right_side(momentum, L, N)
            no_of_momentum_slices += 1
    return sumrule / no_of_momentum_slices


if __name__ == "__main__":
    import rho_form_factor as rff
    import copy
    import lieb_liniger_state as lls

    L = N = 9
    data = {}
    rstate = lls.lieb_liniger_state(1, L, N)
    rstate.calculate_all()
    bethe_numbers = copy.copy(rstate.Is)
    for k in range(1000):
        # bethe_numbers = lls.generate_bethe_numbers(N, list(rstate.Is))
        bethe_numbers[-1] += 1
        lstate = lls.lieb_liniger_state(1, L, N, bethe_numbers)
        lstate.calculate_all()
        lstate.ff = rff.calculate_normalized_form_factor(lstate, rstate)
        integer_momentum = lstate.integer_momentum
        if integer_momentum in data:
            data[integer_momentum].append(lstate)
        else:
            data[integer_momentum] = [lstate]

    # for k in sorted(data):
    #     left_side_value = left_side(data[k], rstate.energy)
    #     right_side_value = right_side(k, N, N)
    #     print(k, left_side_value / right_side_value, "\n")
    print(compute_average_sumrule(data, rstate.energy, N, N, print_all=True))
