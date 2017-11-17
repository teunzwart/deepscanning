#include "constants.hpp"
#include "lieb_liniger_state.hpp"

#include <Eigen/Dense>

#include <ctime>
#include <iomanip>
#include <iostream>
    

int main() {

    clock_t time_a = clock();
    for (int n = 0; n < 10; n++) {
        Eigen::MatrixXd bethe_numbers = generate_bethe_numbers(100);
        lieb_liniger_state ll_state(1, 100, 100, bethe_numbers);
        ll_state.find_rapidities();
        // for (auto rap: ll_state.lambdas) {
            // std::cout << std::setw(8) << std::setprecision(4) << rap << " ";
        // }
        // std::cout << std::endl;
    }

    clock_t time_b = clock();

    std::cout << "time: " << (time_b - time_a) / (double)CLOCKS_PER_SEC << std::endl;
    // lieb_liniger_state ll_state(1, 100, 100);
    // ll_state.find_rapidities(true);
    // for (auto rap: ll_state.lambdas) {
    //     std::cout << std::setw(6) << std::setprecision(5) << rap << " ";
    // }
    // std::cout << std::endl;
}

