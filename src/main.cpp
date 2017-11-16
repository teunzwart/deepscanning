#include "lieb_liniger_state.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    for (int n = 0; n < 100; n++) {
        std::vector<double> bethe_numbers = generate_bethe_numbers(10);
        lieb_liniger_state ll_state(1, 100, 10, bethe_numbers);
        ll_state.calculate_rapidities();
        for (auto rap: ll_state.lambdas) {
            std::cout << std::setw(8) << std::setprecision(4) << rap << " ";
        }
        std::cout << std::endl;
    }
    // lieb_liniger_state ll_state(1, 100, 100);
    // ll_state.calculate_rapidities();
    // for (auto rap: ll_state.lambdas) {
    //     std::cout << std::setw(8) << std::setprecision(4) << rap << " ";
    // }
    // std::cout << std::endl;
    // generate_bethe_numbers(10);
}
