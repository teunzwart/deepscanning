#ifndef MACHINE_LEARNING_H
#define MACHINE_LEARNING_H

#include <Eigen/Dense>


Eigen::VectorXd guess_rapidities(Eigen::VectorXd bethe_numbers);
void update_neural_net(Eigen::VectorXd bethe_numbers, Eigen::VectorXd converged_rapidities);

#endif // MACHINE_LEARNING_H
