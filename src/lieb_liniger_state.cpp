#include "constants.hpp"
#include "lieb_liniger_state.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>


lieb_liniger_state::lieb_liniger_state(double new_c, double new_L,
                                       int new_N)
    : c(new_c),
      L(new_L),
      N(new_N),
      Is(static_cast<std::vector<double>::size_type>(new_N)),
      lambdas(static_cast<std::vector<double>::size_type>(new_N)),
      gaudin_matrix(new_N, new_N)
{
    generate_gs_bethe_numbers();
}

lieb_liniger_state::lieb_liniger_state(double new_c, double new_L,
                                       int new_N,
                                       std::vector<double> new_Is)
    : c(new_c),
      L(new_L),
      N(new_N),
      Is(new_Is),
      lambdas(static_cast<std::vector<double>::size_type>(new_N)),
      gaudin_matrix(new_N, new_N)
{
}

/**
 * Generate the Bethe numbers of the N particle Lieb-Liniger ground state.
 */
void lieb_liniger_state::generate_gs_bethe_numbers() {
    if (N % 2 == 0) {
        std::generate(Is.begin(), Is.end(),
                      [this, n=0]() mutable {return -N/2 + 0.5 + n++;});
    } else {
        std::generate(Is.begin(), Is.end(),
                      [this, n=0]() mutable {return -N/2 + n++;});
    }
}

/**
 * Calculate the rapidities of the N particle Lieb-Liniger state.
 *
 * We use a multi-dimensional Newton method, where for a function F
 * we have \f$J_F(x_n) (x_{n+1} - x_n) = - F(x_n)\f$, with \f$J_\fF$ the 
 * Jacobian, which in this case is the Gaudin matrix. 
 */
void lieb_liniger_state::calculate_rapidities() {
    // Initial bad guess for the rapidities.
    std::transform(Is.begin(), Is.end(), lambdas.begin(),
                   [this](double I){return 2 * PI / L * I;});

    for (int t = 0; t < 20; t++) {
        // Calculate the RHS values of the Bethe equations
        Eigen::MatrixXd rhs_bethe_equations = Eigen::MatrixXd(N, 1);
        rhs_bethe_equations.setZero();
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                rhs_bethe_equations(j) += atan((lambdas[j] - lambdas[k]) / c);
            }
            rhs_bethe_equations(j) *= 1 / L;
            rhs_bethe_equations(j) += lambdas[j] - 2 * PI / L * Is[j];
        }
        // std::cout << rhs_bethe_equations << std::endl;

        calculate_gaudin_matrix();
        // std::cout << gaudin_matrix << std::endl;
        Eigen::MatrixXd delta_lambda = gaudin_matrix.fullPivLu().solve(rhs_bethe_equations * -1);
        // std::cout << delta_lambda << std::endl;

        for (int p = 0; p < N; p++) {
            lambdas[p] += delta_lambda(p);
        }
        for (auto l: lambdas) {
            std::cout << std::setprecision(16) << l << ", ";
        }
        std::cout << std::endl;
    }
}

/**
 * Calculate the Gaudin matrix of a Lieb-Liniger state.
 */
void lieb_liniger_state::calculate_gaudin_matrix() {
    for (std::vector<double>::size_type j = 0; j < lambdas.size(); j++) {
        for (std::vector<double>::size_type k = 0; k < lambdas.size(); k++) {
            if (j == k) {
                double kernel_sum = L;
                for(auto lambda: lambdas) {
                    kernel_sum += kernel(lambdas[j] - lambda, c);
                }
                gaudin_matrix(j, k) = kernel_sum;
            } else {
                gaudin_matrix(j, k) = -kernel(lambdas[j] - lambdas[k], c);
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

