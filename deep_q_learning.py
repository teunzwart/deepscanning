"""
Based in part on http://outlace.com/rlpart3.html

I used this as a way to understand Q-learning, 
and then wrote my version based on this.
"""

import sys

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from scipy.misc import comb

import lieb_liniger_state as lls
import rho_form_factor as rff
from sum_rule import compute_average_sumrule, left_side, right_side
from utils import map_to_entire_space, map_to_bethe_numbers, get_allowed_indices, select_action


def neural_net(N_world):
    model = Sequential()
    model.add(Dense(units=int(N_world**1.5), kernel_initializer='lecun_uniform', activation='tanh', input_dim=N_world))
    model.add(Dense(units=int(N_world**1.5), kernel_initializer='lecun_uniform', activation='tanh'))
    model.add(Dense(units=N_world**2, kernel_initializer='lecun_uniform', activation='tanh'))
    model.compile(loss='mse', optimizer=RMSprop())
    model.summary()
    return model


def get_sumrule_reward(lstate, rstate):
    return (lstate.energy - rstate.energy) * np.abs(rff.rho_form_factor(lstate, rstate))**2


def get_reward_for_close_states(ff, lstate, rstate, N_world):
    if np.abs(ff) > 0.00001:
        return np.abs(ff)**0.1
    elif distance_to_rstate(lstate, rstate) < N_world**0.5:
        return 1 / distance_to_rstate(lstate, rstate)**0.1
    else:
        return -1


def get_form_factor_reward(lstate, rstate):
    return np.tanh(np.log1p(np.abs(rff.rho_form_factor(lstate, rstate))**2))

def get_reward_at_final_step(dsf_data, n, no_of_steps, c, L, N, I_max, N_world, rstate):
    # Quite logical that this does not perform well since it puts all reward into single place, which means learning (if any) is very slow.
    if n == no_of_steps:
        return compute_average_sumrule(dsf_data, rstate.energy, L, N, I_max, N_world)
    else:
        return 0


def get_partial_sumrule_reward_at_every_step(dsf_data, c, L, N, lstate, rstate):
    if lstate.integer_momentum != 0:
        return left_side(dsf_data[lstate.integer_momentum], rstate.energy) / right_side(lstate.integer_momentum, L, N)
    else:
        return 0


def get_full_sumrule_reward_at_every_step(dsf_data, L, N, I_max, N_world, rstate):
    return compute_average_sumrule(dsf_data, rstate.energy, L, N, I_max, N_world)


def get_relative_contribution_reward(dsf_data, L, N, I_max, N_world, N_states, rstate):
    return get_full_sumrule_reward_at_every_step(dsf_data, L, N, I_max, N_world, rstate) / N_states


def get_reward_delta_sumrule(dsf_data, L, N, I_max, N_world, prev_sumrule, rstate):
    return get_full_sumrule_reward_at_every_step(dsf_data, L, N, I_max, N_world, rstate) - prev_sumrule


def get_relative_reward_per_slice(dsf_data, c, L, N, k, rstate):
    if k != 0:
        return left_side(dsf_data[k], rstate.energy) / right_side(k, L, N) / len(dsf_data[k])
    else:
        return 0


def distance_to_rstate(lstate, rstate):
    """Calculate the 'distance' between lstate and rstate."""
    return np.sum(np.abs(lstate.Is - rstate.Is))



def epsilon_greedy(qval, state, previously_visited, epsilon, max_I, N_world, N, check_no_of_pairs=True):
    if np.random.random() < epsilon:
        allowed_actions = get_allowed_indices(state, N_world)
        np.random.shuffle(allowed_actions)
        return select_action(allowed_actions, state, previously_visited, max_I, N_world, N, check_no_of_pairs)
    else:
        return select_action(list(zip(*np.unravel_index(qval[0].argsort(), (N_world, N_world)))), state, previously_visited, max_I, N_world, N, check_no_of_pairs)


def q_learning(N_world, I_max, c, L, N, gamma=0.975, alpha=1, epochs=100, epsilon=1, no_of_steps=100, model=None, best_dsf=None, check_no_of_pairs=False):
    # Allow for further training of a given model.
    if not model:
        model = neural_net(N_world)
    rstate = lls.lieb_liniger_state(c, L, N)
    rstate.calculate_all()
    highest_achieved_sumrule = 0
    sums = []
    best_sums = []
    print(f"Size of search space is Choose[N_world, N]={comb(N_world, N):.3e}")
    for i in range(1, epochs + 1):
        dsf_data = {}
        previously_visited_states = []
        state = np.array(map_to_entire_space(rstate.Is, I_max), dtype=np.int)
        previously_visited_states.append(list(state))
        previous_sumrule = 0
        for n in range(1, no_of_steps + 1):
            Q = model.predict(state.reshape(1, -1), batch_size=1)
            new_state, action = epsilon_greedy(Q, state, previously_visited_states, epsilon, I_max, N_world, N, check_no_of_pairs=check_no_of_pairs)
            previously_visited_states.append(list(new_state))

            new_lstate = lls.lieb_liniger_state(c, L, N, map_to_bethe_numbers(new_state, I_max))
            new_lstate.calculate_all()
            new_lstate.ff = rff.rho_form_factor(new_lstate, rstate)

            if new_lstate.integer_momentum in dsf_data.keys():
                dsf_data[new_lstate.integer_momentum].append(new_lstate)
            else:
                dsf_data[new_lstate.integer_momentum] = [new_lstate]

            # reward = get_sumrule_reward(new_lstate, rstate)
            reward = get_reward_for_close_states(new_lstate.ff, new_lstate, rstate, N_world)
            # reward = get_form_factor_reward(new_lstate, rstate)

            # reward = get_reward_at_final_step(dsf_data, i, no_of_steps, c, L, N, I_max, N_world, rstate)
            # reward = get_partial_sumrule_reward_at_every_step(dsf_data, c, L, N, new_lstate, rstate)

            # reward = get_full_sumrule_reward_at_every_step(dsf_data, c, L, N, I_max, N_world, rstate)
            # reward = get_relative_contribution_reward(dsf_data, L, N, I_max, N_world, n, rstate)
            # reward = get_reward_delta_sumrule(dsf_data, L, N, I_max, N_world, previous_sumrule, rstate)
            # reward = get_relative_reward_per_slice(dsf_data, c, L, N, I_max, N_world, new_lstate.integer_momentum, rstate)


            new_Q = model.predict(new_state.reshape(1, -1), batch_size=1)
            _, new_action = select_action(list(zip(*np.unravel_index(new_Q[0].argsort(), (N_world, N_world)))), state, previously_visited_states, I_max, N_world, N, check_no_of_pairs=False)
            new_best_action = np.ravel_multi_index(new_action, (N_world, N_world))
            new_max_Q = new_Q[0][new_best_action]

            y = np.zeros((1, N_world * N_world))
            y[:] = Q[:]

            if n == no_of_steps:
                update = reward
            else:
                update = (reward + gamma * new_max_Q)

            y[0][new_best_action] = (1 - alpha) * y[0][new_best_action] + alpha * update
            # A batch size 1 makes a huge positive difference in learning performance (probably because there is less overfitting to the single data point).
            model.fit(state.reshape(1, -1), y, batch_size=1, verbose=0)

            state = new_state

            prev_sumrule = compute_average_sumrule(dsf_data, rstate.energy, L, N, I_max, N_world, print_all=False)

            sys.stdout.write(f"epoch: {i:{len(str(epochs))}}, n={n:{len(str(no_of_steps))}}, current sumrule: {compute_average_sumrule(dsf_data, rstate.energy, L, N, I_max, N_world, print_all=False):.10f}, best sumrule: {highest_achieved_sumrule:.10f}\r")
            sys.stdout.flush()

        if epsilon > 0.1:
            epsilon -= 1 / epochs

        ave_sum_rule = compute_average_sumrule(dsf_data, rstate.energy, L, N, I_max, N_world, print_all=False)
        sums.append(ave_sum_rule)
        print(f"epoch: {i:{len(str(epochs))}}, n={n:{len(str(no_of_steps))}}, current sumrule: {ave_sum_rule:.10f}, best sumrule: {highest_achieved_sumrule:.10f}")
        if ave_sum_rule > highest_achieved_sumrule:
            highest_achieved_sumrule = ave_sum_rule
            best_dsf = dsf_data
        best_sums.append(highest_achieved_sumrule)

    return model, best_dsf, sums, best_sums

