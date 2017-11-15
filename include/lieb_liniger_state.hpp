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
    double c;
    double L;
    int N;
    std::vector<double> Is;
    std::vector<double> lambdas;
    Eigen::MatrixXd gaudin_matrix;
    
    
    lieb_liniger_state(double new_c, double new_L, int new_N);
    lieb_liniger_state(double new_c, double new_L, int new_N,
                       std::vector<double> new_Is);

    void calculate_rapidities();
    void calculate_gaudin_matrix();
    static double kernel(double k, double c);


private:
    void generate_gs_bethe_numbers();
};


#endif // LIEB_LINIGER_STATE_H
