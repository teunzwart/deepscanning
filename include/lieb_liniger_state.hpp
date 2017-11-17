#ifndef LIEB_LINIGER_STATE_H
#define LIEB_LINIGER_STATE_H

#include <Eigen/Dense>

/**
 * Represent a Lieb-Liniger Bethe state.
 */
class lieb_liniger_state
{
public:
    double c; ///< Interaction strength.
    double L; ///< System size.
    int N; ///< Number of particles
    Eigen::MatrixXd Is; ///< Bethe numbers
    Eigen::MatrixXd lambdas; ///< Rapidities
    Eigen::MatrixXd gaudin_matrix; ///< The Gaudin matrix of the state.
    
    
    lieb_liniger_state(double new_c, double new_L, int new_N);
    lieb_liniger_state(double new_c, double new_L, int new_N,
                       Eigen::MatrixXd new_Is);

    void calculate_rapidities_newton();
    void calculate_gaudin_matrix();
    static double kernel(double k, double c);

    void find_rapidities(bool use_machine_learning=false);

private:
    void generate_gs_bethe_numbers();
};

Eigen::MatrixXd generate_bethe_numbers(int N);


#endif // LIEB_LINIGER_STATE_H
