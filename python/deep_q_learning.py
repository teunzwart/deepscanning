"""Based in part on http://outlace.com/rlpart3.html"""

import sys

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

import lieb_liniger_state as lls
import rho_form_factor as rff
from sum_rule import compute_average_sumrule
from utils import map_to_entire_space, map_to_bethe_numbers, get_valid_random_action, is_valid_action, array_to_list


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


def change_state(state, action):
    state[action[0]] -= 1
    state[action[1]] += 1
    return state


def get_reward(lstate, rstate):
    return (lstate.energy - rstate.energy) * np.abs(rff.calculate_normalized_form_factor(lstate, rstate))**2


def getReward(ff, lstate, rstate, N_world):
    if np.abs(ff) > 0.00001:
        return np.abs(ff)**0.1
    elif distance_to_rstate(lstate, rstate) < N_world**0.5:
        return distance_to_rstate(lstate, rstate)**0.1
    else:
        return -1


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
            # print(a)
            # print(state)
            # print(is_valid_action(state, a[1], N_world))
            if is_valid_action(state, a[1], N_world):
                action = a[1]
                # sys.stdout.write(f"{state}, {a}\r")
                # sys.stdout.flush()
                new_state = change_state(state, action)
                if list(new_state) not in previously_visited:
                    return new_state, action


def q_learning(N_world, max_I, L, N, gamma=0.975, epochs=100, epsilon=1, no_of_steps=100):
    model = neural_net(N_world)
    rstate = lls.lieb_liniger_state(1, L, N)
    rstate.calculate_all()
    highest_achieved_sumrule = 0
    sums = []
    best_dsf = None
    for i in range(epochs):
        dsf_data = {}
        previously_visited = []
        state = map_to_entire_space(rstate.Is, max_I)
        for n in range(no_of_steps + 1):
            Q = model.predict(state.reshape(1, -1))
            new_state, action = epsilon_greedy(Q, state, previously_visited, epsilon, N_world)
            previously_visited.append(list(new_state))
            new_lstate = lls.lieb_liniger_state(1, N, L, map_to_bethe_numbers(new_state, max_I))
            new_lstate.calculate_all()
            new_lstate.ff = rff.calculate_normalized_form_factor(new_lstate, rstate)
            # reward = get_reward(new_lstate, rstate)
            reward = getReward(new_lstate.ff, new_state, map_to_entire_space(rstate.Is, max_I), N_world)

            if new_lstate.integer_momentum in dsf_data.keys():
                dsf_data[new_lstate.integer_momentum].append(new_lstate)
            else:
                dsf_data[new_lstate.integer_momentum] = [new_lstate]

            new_Q = model.predict(new_state.reshape(1, -1))
            # TODO: This should be the largest allowed Q value.
            new_max_Q = np.argmax(new_Q)

            y = np.zeros((N_world * N_world))
            y[:] = Q[:]

            if n == epochs - 1:
                update = reward
            else:
                update = reward + gamma * new_max_Q

            y[np.ravel_multi_index(action, (N_world, N_world))] = update
            model.fit(state.reshape(1, -1), y.reshape(1, -1), verbose=0)

            state = new_state

            sys.stdout.write(f"epoch: {i:4}, n={n:4}, {compute_average_sumrule(dsf_data, rstate.energy, N, L, False):.10f}, {highest_achieved_sumrule:.10f} \r")
            sys.stdout.flush()

        if epsilon > 0.1:
            epsilon -= 1 / epochs

        ave_sum_rule = compute_average_sumrule(dsf_data, rstate.energy, N, L, False)
        sums.append(ave_sum_rule)
        print(f"epoch: {i:4}, n={n:4}, {ave_sum_rule:.10f}, {highest_achieved_sumrule:.10f}")
        if ave_sum_rule > highest_achieved_sumrule:
            highest_achieved_sumrule = ave_sum_rule
            best_dsf = dsf_data

    print("highest", highest_achieved_sumrule)
    print(sums)
    return best_dsf


if __name__ == "__main__":
    # N_world = 2 * max_I + 1
    q_learning(41, 20, 9, 9)
