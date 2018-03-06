"""Based in part on http://outlace.com/rlpart3.html"""

import sys

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from scipy.misc import comb

import lieb_liniger_state as lls
import rho_form_factor as rff
from sum_rule import compute_average_sumrule, left_side, right_side
from utils import map_to_entire_space, map_to_bethe_numbers, get_valid_random_action, is_valid_action, array_to_list, get_largest_allowed_Q_value, change_state


def neural_net(N_world):
    model = Sequential()
    model.add(Dense(units=N_world, activation='relu', kernel_initializer='lecun_uniform', input_dim=N_world))
    model.add(Dense(units=int(N_world**1.5), kernel_initializer='lecun_uniform', activation='softmax'))
    model.add(Dense(units=N_world**2, kernel_initializer='lecun_uniform', activation='relu'))
    model.compile(loss='mse', optimizer=RMSprop())
    model.summary()
    return model

    # model = Sequential()
    # model.add(Dense(N_world*5, kernel_initializer='lecun_uniform', input_shape=(N_world,), activation="relu"))
    # model.add(Dense(N_world*5, kernel_initializer='lecun_uniform', activation="relu"))
    # model.add(Dense(N_world**2, kernel_initializer='lecun_uniform', activation="tanh"))
    # model.compile(loss='mse', optimizer=RMSprop())
    # model.summary()
    # return model


def get_formfactor_reward(lstate, rstate):
    return (lstate.energy - rstate.energy) * np.abs(rff.calculate_normalized_form_factor(lstate, rstate))**2


def get_reward_for_large_formfactors(ff, lstate, rstate, N_world):
    if np.abs(ff) > 0.00001:
        return np.abs(ff)**0.1
    elif distance_to_rstate(lstate, rstate) < N_world**0.5:
        return distance_to_rstate(lstate, rstate)**0.1
    else:
        return 0


def get_reward_at_final_step(dsf_data, n, no_of_steps, L, N):
    # Quite logical that this does not perform well since it puts all reward into single place, which means learning (if any) is very slow.
    if n == no_of_steps:
        state = lls.lieb_liniger_state(1, L, N)
        state.calculate_all()
        return compute_average_sumrule(dsf_data, state.energy, L, N,)
    else:
        return 0


def get_partial_sumrule_reward_at_every_step(dsf_data, n, no_of_steps, L, N):
    state = lls.lieb_liniger_state(1, L, N)
    state.calculate_all()
    if state.momentum != 0:
        return left_side(states, state.ref_energy) / right_side(state.momentum, L, N)
    else:
        return 0


def get_full_sumrule_reward_at_every_step(dsf_data, L, N):
    state = lls.lieb_liniger_state(1, L, N)
    state.calculate_all()
    return compute_average_sumrule(dsf_data, state.energy, L, N,)


def distance_to_rstate(lstate, rstate):
    """Calculate the 'distance' between lstate and rstate."""
    distance = 0
    for i, n in enumerate(np.argwhere(np.array(rstate) == 1)):
        distance += np.abs(n[0] - np.where(np.array(lstate) == 1)[0][i])
    return distance


def epsilon_greedy(qval, state, previously_visited, epsilon, N_world):
    if np.random.random() < epsilon:
        while True:
            action = get_valid_random_action(state, N_world)
            new_state = change_state(state, action)
            if list(new_state) not in previously_visited:
                return new_state, action
    else:
        for a in array_to_list(qval.reshape(N_world, N_world)):
            if is_valid_action(state, a[1], N_world):
                action = a[1]
                new_state = change_state(state, action)
                if list(new_state) not in previously_visited:
                    return new_state, action


def q_learning(N_world, I_max, L, N, gamma=0.975, alpha=1, epochs=100, epsilon=1, no_of_steps=100):
    model = neural_net(N_world)
    rstate = lls.lieb_liniger_state(1, L, N)
    rstate.calculate_all()
    highest_achieved_sumrule = 0
    sums = []
    best_sums = []
    best_dsf = None
    print(f"Size of search space is Choose[N_world, N]={comb(N_world, N):.3e}")
    for i in range(1, epochs + 1):
        dsf_data = {}
        previously_visited_states = []
        state = map_to_entire_space(rstate.Is, I_max)
        for n in range(1, no_of_steps + 1):
            Q = model.predict(state.reshape(1, -1), batch_size=1)
            new_state, action = epsilon_greedy(Q, state, previously_visited_states, epsilon, N_world)
            previously_visited_states.append(list(new_state))

            new_lstate = lls.lieb_liniger_state(1, N, L, map_to_bethe_numbers(new_state, I_max))
            new_lstate.calculate_all()
            new_lstate.ff = rff.calculate_normalized_form_factor(new_lstate, rstate)

            # reward = get_formfactor_reward(new_lstate, rstate)
            reward = get_reward_for_large_formfactors(new_lstate.ff, new_state, map_to_entire_space(rstate.Is, I_max), N_world)
            # reward = get_partial_sumrule_reward_at_every_step(dsf_data, n, no_of_steps, L, N)
            # reward = get_reward_at_final_step(dsf_data, n, no_of_steps, L, N)
            # reward = get_full_sumrule_reward_at_every_step(dsf_data, L, N)

            if new_lstate.integer_momentum in dsf_data.keys():
                dsf_data[new_lstate.integer_momentum].append(new_lstate)
            else:
                dsf_data[new_lstate.integer_momentum] = [new_lstate]

            new_Q = model.predict(new_state.reshape(1, -1), batch_size=1)
            # TODO: This should be the largest allowed Q value.
            # new_max_Q = np.max(new_Q)
            new_max_Q = get_largest_allowed_Q_value(new_Q, new_state, previously_visited_states, N_world)

            y = np.zeros((1, N_world * N_world))
            y[:] = Q[:]

            if n == no_of_steps:
                update = alpha * reward
            else:
                update = alpha * (reward + gamma * new_max_Q)

            y[0][np.ravel_multi_index(action, (N_world, N_world))] = (1 - alpha) * y[0][np.ravel_multi_index(action, (N_world, N_world))] + alpha * update
            # A batch size 1 makes a huge positive difference in learning performance (probably because there is less overfitting to the single data point).
            model.fit(state.reshape(1, -1), y, batch_size=1, verbose=0)

            state = new_state

            sys.stdout.write(f"epoch: {i:{len(str(epochs))}}, n={n:{len(str(no_of_steps))}}, current sumrule: {compute_average_sumrule(dsf_data, rstate.energy, L, N, False):.10f}, best sumrule: {highest_achieved_sumrule:.10f} \r")
            sys.stdout.flush()

        if epsilon > 0.1:
            epsilon -= 1 / epochs

        ave_sum_rule = compute_average_sumrule(dsf_data, rstate.energy, L, N, False)
        sums.append(ave_sum_rule)
        print(f"epoch: {i:{len(str(epochs))}}, n={n:{len(str(no_of_steps))}}, current sumrule: {ave_sum_rule:.10f}, best sumrule: {highest_achieved_sumrule:.10f}")
        if ave_sum_rule > highest_achieved_sumrule:
            highest_achieved_sumrule = ave_sum_rule
            best_dsf = dsf_data
        best_sums.append(highest_achieved_sumrule)

    print(sums)
    return model, best_dsf, sums, best_sums


if __name__ == "__main__":
    # N_world = 2 * I_max + 1
    q_learning(41, 20, 9, 9)
