import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import keras
from keras.models import load_model

from visualize import *
from deep_q_learning import *
from lieb_liniger_state import *
from sum_rule import *
from utils import map_to_entire_space, map_to_bethe_numbers, get_valid_random_action, change_state, measure_with_error
from calculate_dsf import *

L = N = 5
Imax = 12
c = 1
rand = False
check_no_of_pairs_in_training = False
check_no_of_pairs_in_evaluation = True


N_world = 2 * Imax + 1

def experiment(no_of_iterations):
    saturation_histories = []
    sumrules_saturations = []
    mid_points = []
    coefficients = []
    steps_to_convergence = []
    for iteration in range(no_of_iterations):

        if not rand:
            model, best_dsf, sums, best_sums = q_learning(N_world, Imax, c, L, N, alpha=0.1, 
                                                          no_of_steps=500, epochs=25, epsilon=0.1, check_no_of_pairs=check_no_of_pairs_in_training)
        else:
            model = None

        dsfs, saturation_history, form_factors, steps = dsf_scan(model, N_world, Imax, c, L, N, 
                                                                 max_no_of_steps=3000, 
                                                                 prefered_sumrule_saturation=0.83, 
                                                                 is_random=rand, 
                                                                 check_no_of_pairs=check_no_of_pairs_in_evaluation)

        saturation_histories.append(saturation_history)
        steps_to_convergence.append(steps)

        refstate = lls.lieb_liniger_state(c, L, N)
        refstate.calculate_all()
        average_sum_rule = compute_average_sumrule(dsfs, refstate.energy, L, N, Imax, N_world, print_all=False)
        print("achieved sum rule", average_sum_rule, "steps:", steps)
        sumrules_saturations.append(average_sum_rule)

        if iteration == list(range(no_of_iterations))[-1]:
            save_ff = True
        else:
            save_ff = False
        mid, coeff = visualize_form_factor_sizes(form_factors, save=save_ff)
        mid_points.append(mid)
        coefficients.append(coeff)

        # visualize_sumrule_per_contributing_state(dsfs, refstate.energy, L, N, xlim=[-Imax-1, Imax+1], save=False)

        if model:
            lstate = map_to_entire_space(lieb_liniger_state(c, L, N).Is, Imax)
            visualize_state(lstate, save=False)

            pred = model.predict(lstate.reshape(1,-1)).reshape(N_world, N_world)

            visualize_q_function(pred, save=False)
            visualize_q_function(pred, generate_overlay(pred, "lowest", 30, N_world), save=False)
            visualize_q_function(pred, generate_overlay(pred, "highest", 30, N_world), save=False)

    visualize_saturation_history(saturation_histories, save=True)
    measure_with_error(np.array(mid_points), "mid")
    measure_with_error(np.array(coefficients), "coeff",)
    measure_with_error(np.array(sumrules_saturations), "sum rule")
    measure_with_error(np.array(steps_to_convergence), "steps to convergence")


