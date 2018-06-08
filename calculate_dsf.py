import sys

from scipy.misc import comb

import lieb_liniger_state as lls
from deep_q_learning import epsilon_greedy
from utils import map_to_entire_space, map_to_bethe_numbers
import rho_form_factor as rff
from sum_rule import compute_average_sumrule


def dsf_scan(model, N_world, I_max, L, N, max_no_of_steps=10000, prefered_sumrule_saturation=0.9, is_random=False, check_no_of_pairs=False):
    rstate = lls.lieb_liniger_state(1, L, N)
    rstate.calculate_all()
    print(f"Size of search space is Choose[N_world, N]={comb(N_world, N):.3e}")
    dsf_data = {}
    previously_visited_states = []
    saturation_history = []
    form_factors = []
    state = map_to_entire_space(rstate.Is, I_max)
    previously_visited_states.append(list(state))
    if is_random:
        epsilon = 1
    else:
        epsilon = 0.1
    for n in range(1, max_no_of_steps + 1):
        Q = model.predict(state.reshape(1, -1), batch_size=1)
        new_state, action = epsilon_greedy(Q, state, previously_visited_states, epsilon, I_max, N_world, N, check_no_of_pairs=check_no_of_pairs)
        previously_visited_states.append(list(new_state))

        new_lstate = lls.lieb_liniger_state(1, N, L, map_to_bethe_numbers(new_state, I_max))
        new_lstate.calculate_all()
        new_lstate.ff = rff.rho_form_factor(new_lstate, rstate)
        form_factors.append(new_lstate.ff)

        if new_lstate.integer_momentum in dsf_data.keys():
            dsf_data[new_lstate.integer_momentum].append(new_lstate)
        else:
            dsf_data[new_lstate.integer_momentum] = [new_lstate]

        state = new_state

        sum_rule_saturation = compute_average_sumrule(dsf_data, rstate.energy, L, N, I_max, N_world, print_all=False)
        saturation_history.append(sum_rule_saturation)

        sys.stdout.write(f"n={n:{len(str(max_no_of_steps))}}, current sumrule: {sum_rule_saturation:.10f} \r")

        if sum_rule_saturation > prefered_sumrule_saturation:
            return dsf_data, saturation_history, form_factors

    return dsf_data, saturation_history, form_factors
