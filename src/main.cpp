#include "constants.hpp"
#include "lieb_liniger_state.hpp"

#include <Eigen/Dense>

#include <ctime>
#include <iomanip>
#include <iostream>
    

int main() {
    for (int n = 0; n < 1; n++) {
        Eigen::VectorXd bethe_numbers = generate_bethe_numbers(10);
        std::cout << bethe_numbers << std::endl;
        lieb_liniger_state llstate(1, 100, 10, bethe_numbers);
        llstate.find_rapidities();
        std::cout << llstate.lambdas << std::endl;
    }
    
    lieb_liniger_state llstate2(1, 100, 9);
    llstate2.find_rapidities(true);
    std::cout << llstate2.lambdas << std::endl;
}

