import lieb_liniger_state as lls
import psi_form_factor as pff

import numpy as np


def right_side(k, N, L):
    return 4 * np.pi**2 * N / L**4 * k**2 

def left_side(list_of_states, ref_energy):
    sumrule_sum = 0
    for state in list_of_states:
        print("energy diff", (state.energy - ref_energy))
        print(state.ff**2)
        sumrule_sum += (state.energy - ref_energy) * np.real(state.ff**2)

    return sumrule_sum



if __name__ == "__main__":
    import copy
    N = 5
    data = {}
    rstate = lls.lieb_liniger_state(1, N, N)
    rstate.calculate_all()
    bethe_numbers = copy.copy(rstate.Is)
    orig_bethe = copy.copy(rstate.Is)
    for k in range(100):
        # bethe_numbers = lls.generate_bethe_numbers(N, list(rstate.Is))
        bethe_numbers[-1] += 1
        lstate = lls.lieb_liniger_state(1, N, N, bethe_numbers)
        lstate.calculate_all()
        lstate.ff = pff.calculate_normalized_form_factor(lstate, rstate)
        integer_momentum = lstate.integer_momentum
        if integer_momentum in data:
            data[integer_momentum].append(lstate)
        else:
            data[integer_momentum] = [lstate]

    for k in sorted(data):
        left_side_value = left_side(data[k], rstate.energy)
        right_side_value = right_side(k, N, N)
        print(k, left_side_value / right_side_value, "\n")
