#ifndef LIEB_LINIGER_STATE_H
#define LIEB_LINIGER_STATE_H

#include <vector>

class lieb_liniger_state
{
public:
    double c;
    double L;
    int N;
    std::vector<double> Is;
    std::vector<double> lambdas;
    
    lieb_liniger_state(double new_c, double new_L, int new_N);
    lieb_liniger_state(double new_c, double new_L, int new_N,
                       std::vector<double> new_Is);

    void generate_gs_bethe_numbers();
};

#endif // LIEB_LINIGER_STATE_H
