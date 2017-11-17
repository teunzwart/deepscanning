#include "constants.hpp"
#include "lieb_liniger_state.hpp"
#include "machine_learning.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

/**
 * Construct a Bethe state with Bethe numbers of the ground state.
 */
lieb_liniger_state::lieb_liniger_state(double new_c, double new_L,
                                       int new_N)
    : c(new_c),
      L(new_L),
      N(new_N),
      Is(new_N),
      lambdas(new_N),
      gaudin_matrix(new_N, new_N)
{
    generate_gs_bethe_numbers();

}

/**
 * Construct a Bethe state with arbitrary Bethe numbers.
 */
lieb_liniger_state::lieb_liniger_state(double new_c, double new_L,
                                       int new_N,
                                       Eigen::VectorXd new_Is)
    : c(new_c),
      L(new_L),
      N(new_N),
      Is(new_Is),
      lambdas(new_N),
      gaudin_matrix(new_N, new_N)
{
}

/**
 * Generate the Bethe numbers of the N particle Lieb-Liniger ground state.
 */
void lieb_liniger_state::generate_gs_bethe_numbers() {
    if (N % 2 == 0) {
        Is = Eigen::VectorXd::LinSpaced(N, 1, N) - Eigen::VectorXd::Constant(N,  N/2);
        Is -= Eigen::VectorXd::Constant(N, 0.5);
    } else {
        Is = Eigen::VectorXd::LinSpaced(N, -(N-1)/2, (N-1)/2);
    }
}

void lieb_liniger_state::find_rapidities(bool use_machine_learning) {
    if (!use_machine_learning) {
        // Initial bad guess for the rapidities.
        lambdas = 2 * PI / L * Is;
        calculate_rapidities_newton();
    } else {
        lambdas = guess_rapidities(Is);
        calculate_rapidities_newton();
        update_neural_net(Is, lambdas);
    }
}

/**
 * Calculate the rapidities of the N particle Lieb-Liniger state.
 *
 * We use a multi-dimensional Newton method, where for a function F
 * we have \f$J_F(x_n) (x_{n+1} - x_n) = - F(x_n)\f$, with \f$J_F\f$ the 
 * Jacobian, which in this case is the Gaudin matrix.
 *
 * This function is largely inspired by the Find_Rapidities() function
 * for the Lieb-Liniger model found in the ABACUS library by J-S Caux. 
 */
void lieb_liniger_state::calculate_rapidities_newton() {
    for (int no_of_iterations = 0; no_of_iterations < 20; no_of_iterations++) {
        // Calculate the Yang-Yang equation values.
        Eigen::VectorXd rhs_bethe_equations = Eigen::VectorXd(N).setZero();
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                rhs_bethe_equations(j) += 2 * atan((lambdas(j) - lambdas(k) / c));
            }
            rhs_bethe_equations(j) += L * lambdas(j) - 2 * PI * Is(j);
        }

        // Perform a step of the Newton method.
        calculate_gaudin_matrix();
        // Partial LU decomposition may be used sinde the Gaudin matrix is non-singular.
        Eigen::VectorXd delta_lambda = gaudin_matrix.lu().solve(-rhs_bethe_equations);

        // Calculate the average difference squared of the rapidity changes.
        double diff_square = 0;
        for (int i = 0; i < N; i++) {
            diff_square += delta_lambda(i) * delta_lambda(i) / lambdas(i) * lambdas(i);
        }
        diff_square /= N;

        // Update rapidities.
        for (int p = 0; p < N; p++) {
            lambdas(p) += delta_lambda(p);
        }

        // Rapidities converged.
        if (diff_square < 10e4 * MACHINE_EPS_SQUARE) {
            break;
        }
    }
}

/**
 * Calculate the Gaudin matrix of a Lieb-Liniger state.
 */
void lieb_liniger_state::calculate_gaudin_matrix() {
    for (int j = 0; j < lambdas.size(); j++) {
        for (int k = 0; k < lambdas.size(); k++) {
            if (j == k) {
                double kernel_sum = L;
                for (int kp = 0; kp < N; kp++) {
                    kernel_sum += kernel(lambdas(j) - lambdas(kp), c);
                }
                gaudin_matrix(j, k) = kernel_sum;
            } else {
                gaudin_matrix(j, k) = -kernel(lambdas(j) - lambdas(k), c);
            }
        }
    }
}

/**
 * Calculate the kernel of the Lieb-Liniger model.
 */
double lieb_liniger_state::kernel(double k, double c) {
    return 2 * c / (c * c + k * k);
}

/**
 * Generate a vector of Bethe numbers distributed according to a
 * Gaussian distribution, with only unique entries.
 */
Eigen::VectorXd generate_bethe_numbers(int N) {
    Eigen::VectorXd bethe_numbers = Eigen::VectorXd::Constant(N, -10e7);
    double mean = 0;
    double standard_dev = N * PI; // We assume that excitations still cluster around the ground state.

    std::random_device rd;
    std::mt19937 gen(rd());
 
    std::normal_distribution<> normal_distribution(mean, standard_dev);
    std::uniform_real_distribution<> sign_distribution(-1, 1);

    // TODO: Make sure all entries are unique.
    for (int n = 0; n < N; n++) {
        double random_number = std::round(normal_distribution(gen));
        // std::cout << "rand num " << random_number << std::endl;

        // For N even Bethe numbers are half odd.
        if (N % 2 == 0) {
            bethe_numbers(n) = random_number + std::copysign(1, sign_distribution(gen)) * 0.5;            
        } else {
            bethe_numbers(n) = random_number;
        }
    }

    std::sort(bethe_numbers.data(), bethe_numbers.data() + bethe_numbers.size());
    
    return bethe_numbers;
}
