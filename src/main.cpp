#include "constants.hpp"
#include "lieb_liniger_state.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    // for (int n = 0; n < 1; n++) {
    //     std::vector<double> bethe_numbers = generate_bethe_numbers(100);
    //     lieb_liniger_state ll_state(1, 100, 100, bethe_numbers);
    //     ll_state.find_rapidities();
    //     for (auto rap: ll_state.lambdas) {
    //         std::cout << std::setw(8) << std::setprecision(4) << rap << " ";
    //     }
    //     std::cout << std::endl;
    // }
    lieb_liniger_state ll_state(1, 100, 100);
    ll_state.find_rapidities(true);
    for (auto rap: ll_state.lambdas) {
        std::cout << std::setw(6) << std::setprecision(5) << rap << " ";
    }
    std::cout << std::endl;
}
