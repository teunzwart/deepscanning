#include "machine_learning.hpp"

#include <Eigen/Dense>

/**
 * Make a first guess for the rapidities based on machine learning.
 */
Eigen::VectorXd guess_rapidities(Eigen::VectorXd bethe_numbers) {
    return bethe_numbers;
}

void update_neural_net(Eigen::VectorXd bethe_numbers, Eigen::VectorXd converged_rapidities) {

}
