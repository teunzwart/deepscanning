import pickle
import copy
import sys

import tree_search
import lieb_liniger_state as lls
import sum_rule

import psi_form_factor as pff

import numpy as np
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# define baseline model
def baseline_model(N):
    # create model
    model = Sequential()
    model.add(Dense(N, input_dim=N, activation='relu'))
    model.add(Dense(N, activation='relu'))
    model.add(Dense(N, activation='relu'))
    model.add(Dense(N**2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_allowed_prediction(probs, state):
    """Only give non-zero probabilities to allowed transitions."""
    for i, k in enumerate(state):
        if k == 0:
            for z in range(len(state)):
                probs[i][z] = 0
        if k == 1:
            for z in range(len(state)):
                probs[z][i] = 0
    return probs / np.sum(probs)


def mutate(model, state, history):
    # print("Mutating")
    # print(state)
    probs = get_allowed_prediction(model.predict(state.reshape(1, -1)).reshape(41, 41), state)
    # print("probs", probs)
    sorted_probs = np.flipud(np.sort(probs.reshape(1, -1)[0]))
    # print(sorted_probs)
    for k in sorted_probs:
        new_state = copy.copy(state)
        # print("prob", k)
        pos = np.unravel_index(np.where(probs == k), (41, 41))
        # print(pos[0], pos[1])
        new_state[pos[1][0]] -= 1
        new_state[pos[1][1]] += 1
        # print(new_state)
        if list(new_state) not in history:
            return new_state


def map_to_entire_space(bethe_numbers, max_I):
    """Map to an interval [-max_I, max_I]."""
    # TODO: Fix this.
    if len(bethe_numbers) % 2 == 0:
        sys.exit("Can't handle even state mapping.")
    transformed_bethe_numbers = np.array(bethe_numbers, dtype=np.int) + max_I
    entire_space = np.zeros(2 * max_I + 1)
    for i, _ in enumerate(entire_space):
        if i in transformed_bethe_numbers:
            entire_space[i] = 1
    return entire_space


def map_to_bethe_numbers(entire_space, max_I):
    bethe_numbers = []
    for i, k in enumerate(entire_space):
        if k == 1:
            bethe_numbers.append(i)
    # print(bethe_numbers)
    return np.array(bethe_numbers) - max_I


if __name__ == "__main__":
    data_rem = pickle.load(open("removed_Imax20_N5.pickle", "rb"))
    data_add = pickle.load(open("added_Imax20_N5.pickle", "rb"))

    data = []
    for i, k in enumerate(data_rem):
        data.append({"rstate": k["rstate"], "delta": np.ravel(np.vstack((data_rem[i]["delta"] for _ in range(len(data_rem[i]["delta"])))) * np.vstack((data_add[i]["delta"] for _ in range(len(data_add[i]["delta"])))).T)})

    training_x = np.array([np.array(x["rstate"]) for x in data[:8000]])
    training_y = np.array([x["delta"] for x in data[:8000]])

    test_x = np.array([np.array(x["rstate"]) for x in data[8000:]])
    test_y = np.array([x["delta"] for x in data[8000:]])

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
    model = baseline_model(len(data[0]["rstate"]))
    model.fit(training_x, training_y, epochs=10, verbose=1)

    dsf_data = {}
    already_explored = []
    rstate = lls.lieb_liniger_state(1, 5, 5)
    rstate.calculate_all()
    current_state = map_to_entire_space(copy.copy(rstate.Is), 20)
    for k in range(100):
        # print("k", k)
        # print(current_state)
        new_state = mutate(model, current_state, already_explored)
        # print("new_state", new_state)
        already_explored.append(list(current_state))
        lstate = lls.lieb_liniger_state(1, 5, 5, map_to_bethe_numbers(new_state, 20))
        lstate.calculate_all()
        lstate.ff = pff.calculate_normalized_form_factor(lstate, rstate)
        print(lstate.ff)
        
        integer_momentum = lstate.integer_momentum
        if integer_momentum in data:
            dsf_data[integer_momentum].append(lstate)
        else:
            dsf_data[integer_momentum] = [lstate]
        current_state = new_state

    print("sumrule", sum_rule.compute_average_sumrule(dsf_data, rstate.energy, 5, 5))

    print([map_to_bethe_numbers(k, 20) for k in already_explored])
