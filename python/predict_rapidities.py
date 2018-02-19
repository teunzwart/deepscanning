"""Predict rapidities given a set of Bethe numbers."""

import tensorflow as tf
import numpy as np

import lieb_liniger_state as lls

no_of_particles = 10

# Define placeholders to feed mini_batches
X = tf.placeholder(tf.float32, shape=[None, no_of_particles], name='X')
y_ = tf.placeholder(tf.float32, shape=[None, no_of_particles], name='y')

# Find values for W that compute y_data = <x, W>
W = tf.Variable(tf.random_uniform([no_of_particles, no_of_particles], -1.0, 1.0))
b = tf.Variable(np.random.randn(), name="bias")
y = tf.add(tf.matmul(X, W, name='y_pred'), b)

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y_ - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

file_writer = tf.summary.FileWriter('./', sess.graph)


x_data = []
y_data = []

for i in range(256):
    if i % 20 == 0:
        print(i)
    bethe_numbers = lls.generate_bethe_numbers(no_of_particles, [])
    llstate = lls.lieb_liniger_state(1, 100, no_of_particles, bethe_numbers)
    llstate.lambdas = 2 * np.pi / llstate.L * llstate.Is

    llstate.calculate_rapidities_newton()
    x_data.append(llstate.Is)
    y_data.append(llstate.lambdas)

steps = []
errors = []
for step in range(20001):
    sess.run(train, feed_dict={X: x_data[step:step+256], y_: y_data[step:step+256]})

    bethe_numbers = lls.generate_bethe_numbers(no_of_particles, [])
    llstate = lls.lieb_liniger_state(1, 100, no_of_particles, bethe_numbers)
    llstate.lambdas = 2 * np.pi / llstate.L * llstate.Is

    no_of_iterations = llstate.calculate_rapidities_newton()
    x_data.append(llstate.Is)
    y_data.append(llstate.lambdas)
    if step % 200 == 0:
        print("step", step)
        print("no of iterations with naive guess", no_of_iterations)
        classification = sess.run(tf.transpose(y), feed_dict={X: (llstate.Is).reshape(1, no_of_particles)})
        error = np.sum((llstate.lambdas - classification.T.flatten())**2) / no_of_particles
        print("error =", error)
        print("prediction:\n", classification.T.flatten())
        print("exact:\n", llstate.lambdas)

        llstate.lambdas = classification
        print("no of iterations with ML guess", llstate.calculate_rapidities_newton(True), "\n")
        steps.append(step)
        errors.append(error)
