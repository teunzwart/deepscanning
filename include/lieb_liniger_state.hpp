#ifndef LIEB_LINIGER_STATE_H
#define LIEB_LINIGER_STATE_H

#include <Eigen/Dense>

#include <vector>

/**
 * Represent a Lieb-Liniger Bethe state.
 */
class lieb_liniger_state
{
public:
    double c; ///< Interaction strength.
    double L; ///< System size.
    int N; ///< Number of particles
    std::vector<double> Is; ///< Bethe numbers
    std::vector<double> lambdas; ///< Rapidities
    Eigen::MatrixXd gaudin_matrix; ///< The Gaudin matrix of the state.
    
    
    lieb_liniger_state(double new_c, double new_L, int new_N);
    lieb_liniger_state(double new_c, double new_L, int new_N,
                       std::vector<double> new_Is);

    void calculate_rapidities();
    void calculate_gaudin_matrix();
    static double kernel(double k, double c);


private:
    void generate_gs_bethe_numbers();
};

std::vector<double> generate_bethe_numbers(int N);


#endif // LIEB_LINIGER_STATE_H
