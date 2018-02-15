import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

import lieb_liniger_state as lls
import rho_form_factor as rff
from sum_rule import compute_average_sumrule
from utils import map_to_entire_space, map_to_bethe_numbers, get_valid_random_action


def neural_net(N_world):
    model = Sequential()
    model.add(Dense(units=N_world*2, activation='relu', kernel_initializer='lecun_uniform', input_dim=N_world))
    model.add(Dense(units=int(N_world**1.5), kernel_initializer='lecun_uniform', activation='softmax'))
    model.add(Dense(units=N_world**2, kernel_initializer='lecun_uniform', activation='relu'))
    model.compile(loss='mse', optimizer=RMSprop())
    model.summary()
    return model


def change_state(state, action):
    state[action[0]] -= 1
    state[action[1]] += 1
    return state


def get_reward(lstate, rstate):
    return np.abs(rff.calculate_normalized_form_factor(lstate, rstate))**2

def q_learning(N_world, max_I, N, L, gamma=0.9, epochs=100, epsilon=1):
    model = neural_net(N_world)
    rstate = lls.lieb_liniger_state(1, L, N)
    rstate.calculate_all()
    for t in range(epochs):
        print(f"Epoch {t}")
        previously_visited = []
        lstate = lls.lieb_liniger_state(1, L, N)
        lstate.calculate_all()
        for n in range(1000):
            if np.random.random() < epsilon:
                print("random")
                action = get_valid_random_action(lstate.Is, N_world)
            else:
                print("not random")
                Q = model.predict(map_to_entire_space(lstate.Is, max_I).reshape(1, -1))
                action = np.unravel_index(np.argmax(Q), (N_world, N_world))
            print(action, "\n")
            new_state = change_state(map_to_entire_space(lstate.Is, max_I), action)
            reward = get_reward(lls.lieb_liniger_state(1, N, L, map_to_bethe_numbers(new_state, max_I)), rstate)
            print(reward)
            lstate = lls.lieb_liniger_state(1, L, N, map_to_bethe_numbers())

        if epsilon > 0.1:
            epsilon -= 1 / epochs


if __name__ == "__main__":
    q_learning(41, 20, 9, 9)


