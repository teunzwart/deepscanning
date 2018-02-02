import pickle

import tree_search
import lieb_liniger_state as lls

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
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(11, input_dim=11, activation='relu'))
    model.add(Dense(11, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_allowed_removals(probabilities, state):
    """Remove disallowed removals from the probability set."""
    for i, k in enumerate(state):
        if k == 0:
            probabilities[i] = 0
    return probabilities / np.sum(probabilities)

def get_allowed_additions(probabilities, state):
    """Remove disallowed additions from the probability set."""
    for i, k in enumerate(state):
        if k == 1:
            probabilities[i] = 0
    return probabilities / np.sum(probabilities)

if __name__ == "__main__":
    data = pickle.load(open("removed.pickle", "rb"))
    print(len(data))
    training_x = np.array([np.array(x["rstate"]) for x in data[:8000]])
    training_y = np.array([x["delta"] for x in data[:8000]])

    test_x = np.array([np.array(x["rstate"]) for x in data[:8000]])
    test_y = np.array([x["delta"] for x in data[:8000]])
    
    # print(training_x.shape)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
    model = baseline_model()
    model.fit(training_x, training_y, epochs=30, verbose=1)
    print("TESTING\n")
    eval = model.evaluate(test_x, test_y, verbose=1)
    print(eval)
    # results = cross_val_score(estimator, training_x, training_y, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    for i, k in enumerate(test_x[:20]):
        prediction = model.predict(k.reshape(1, -1))
        print("Prediction", np.argmax(prediction), np.argmax(test_y[i]))

    state = lls.lieb_liniger_state(1, 5, 5)
    entire = tree_search.map_to_entire_space(state.Is, 5)
    print(model.predict(entire.reshape(1, -1)))


    data = pickle.load(open("removed.pickle", "rb"))
    print(len(data))
    training_x = np.array([np.array(x["rstate"]) for x in data[:8000]])
    training_y = np.array([x["delta"] for x in data[:8000]])

    test_x = np.array([np.array(x["rstate"]) for x in data[:8000]])
    test_y = np.array([x["delta"] for x in data[:8000]])
    
    # print(training_x.shape)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # evaluate model with standardized dataset
    model = baseline_model()
    model.fit(training_x, training_y, epochs=30, verbose=1)
    print("TESTING\n")
    eval = model.evaluate(test_x, test_y, verbose=1)
    print(eval)
    # results = cross_val_score(estimator, training_x, training_y, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    for i, k in enumerate(test_x[:20]):
        prediction = model.predict(k.reshape(1, -1))
        print("Prediction", np.argmax(prediction), np.argmax(test_y[i]))

    state = lls.lieb_liniger_state(1, 5, 5)
    entire = tree_search.map_to_entire_space(state.Is, 5)
    print(model.predict(entire.reshape(1, -1)))
