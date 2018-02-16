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
    # model.add(Dense(units=int(N_world**1.5), kernel_initializer='lecun_uniform', activation='softmax'))
    model.add(Dense(units=N_world**2, kernel_initializer='lecun_uniform', activation='relu'))
    model.compile(loss='mse', optimizer=RMSprop())
    model.summary()
    return model


def change_state(state, action):
    # print(state, action)
    # print("0", state[action[0]])
    # print("1", state[action[1]])
    try:
        state[action[0]] -= 1
        state[action[1]] += 1
    except (TypeError, IndexError):
        # print(state, action)
        sys.exit()
    # print()
    return state


def get_reward(lstate, rstate):
    return np.abs(rff.calculate_normalized_form_factor(lstate, rstate))**2


def q_learning(N_world, max_I, L, N, gamma=0.9, epochs=100, epsilon=1):
    model = neural_net(N_world)
    rstate = lls.lieb_liniger_state(1, L, N)
    rstate.calculate_all()
    for t in range(epochs):
        print(f"Epoch {t}")
        dsf_data = {}
        previously_visited = []
        lstate = lls.lieb_liniger_state(1, L, N)
        lstate.calculate_all()
        for n in range(100):
            Q = model.predict(map_to_entire_space(lstate.Is, max_I).reshape(1, -1))
            # print(Q.shape)
            if np.random.random() < epsilon:
                # print("random")
                action = get_valid_random_action(map_to_entire_space(lstate.Is, max_I), N_world)
            else:
                # print("not random")
                for a in array_to_list(Q.reshape(N_world, N_world)):
                    if is_valid_action(map_to_entire_space(lstate.Is, max_I), a, N_world):
                        # print("a", a)
                        action = a[1]
                        new_state = change_state(map_to_entire_space(lstate.Is, max_I), action)
                        break
                # action = np.unravel_index(np.argmax(Q), (N_world, N_world))
            # print(action, "\n")
            new_state = change_state(map_to_entire_space(lstate.Is, max_I), action)
            new_lstate = lls.lieb_liniger_state(1, N, L, map_to_bethe_numbers(new_state, max_I))
            try:
                new_lstate.calculate_all()
            except (IndexError, ValueError) as e:
                print(e)
                print(lstate)
                print(new_state)
                print(action)
                sys.exit()
            reward = get_reward(new_lstate, rstate)
            # print("reward", reward)

            new_Q = model.predict(new_state.reshape(1, -1))
            new_max_Q = np.argmax(new_Q)

            y = np.zeros((N_world * N_world))
            y[:] = Q[:]

            update = reward + gamma * new_max_Q
            # print(np.ravel_multi_index(action, (N_world, N_world)))
            y[np.ravel_multi_index(action, (N_world, N_world))] = update

            # print(map_to_entire_space(lstate.Is, max_I).shape)
            # print(y.shape)
            model.fit(map_to_entire_space(lstate.Is, max_I).reshape(1, -1), y.reshape(1, -1), verbose=0)

            lstate = lls.lieb_liniger_state(1, L, N, map_to_bethe_numbers(new_state, max_I))
            lstate.calculate_all()
            previously_visited.append(list(new_state))

            lstate.ff = rff.calculate_normalized_form_factor(lstate, rstate)
            # print(np.abs(lstate.ff)**2)
            # print(lstate.integer_momentum)

            if lstate.integer_momentum in dsf_data.keys():
                dsf_data[lstate.integer_momentum].append(lstate)
            else:
                dsf_data[lstate.integer_momentum] = [lstate]

        print(compute_average_sumrule(dsf_data, rstate.energy, N, L, False))

        if epsilon > 0.1:
            epsilon -= 1 / epochs


if __name__ == "__main__":
    # N_world = 2 * max_I + 1
    q_learning(41, 20, 9, 9)
