"""Predict rapidities given a set of Bethe numbers."""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

import lieb_liniger_state as lls


def neural_net(no_of_particles):
    model = Sequential()
    model.add(Dense(units=no_of_particles, kernel_initializer='lecun_uniform', activation='tanh', input_dim=no_of_particles))
    # model.add(Dense(units=int(no_of_particles**2), kernel_initializer='lecun_uniform', activation='tanh'))
    model.add(Dense(units=int(no_of_particles**3), kernel_initializer='lecun_uniform', activation='tanh'))
    # model.add(Dense(units=int(no_of_particles**2), kernel_initializer='lecun_uniform', activation='tanh'))
    model.add(Dense(units=no_of_particles, kernel_initializer='lecun_uniform'))
    model.compile(loss='mse', optimizer=RMSprop())
    model.summary()
    return model


def main(no_of_particles, no_of_states):
    ground_state = lls.lieb_liniger_state(1, no_of_particles, no_of_particles)
    Is = np.zeros((no_of_states, no_of_particles))
    lambdas = np.zeros((no_of_states, no_of_particles))
    for i in range(no_of_states):
        if i % 1000 == 0:
            print(f"Generated {i}/{no_of_states} states")
        bethe_numbers = lls.generate_bethe_numbers(no_of_particles, ground_state.Is)
        llstate = lls.lieb_liniger_state(1, 100, no_of_particles, bethe_numbers)
        llstate.lambdas = 2 * np.pi / llstate.L * llstate.Is
        no_of_iterations = llstate.calculate_rapidities_newton()
        Is[i] = bethe_numbers
        lambdas[i] = llstate.lambdas

    model = neural_net(no_of_particles)

    history = model.fit(x=Is, y=lambdas, epochs=100, verbose=2, validation_split=0.2)
    print(history.history["val_loss"])

    print(model.predict(Is[0].reshape(1, -1)))
    print(Is[0])
    print(lambdas[0])


if __name__ == "__main__":
    main(no_of_particles=10, no_of_states=20000)
