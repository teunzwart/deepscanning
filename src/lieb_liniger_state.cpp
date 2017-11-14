#include "lieb_liniger_state.hpp"

#include <iostream>
#include <numeric>
#include <vector>


lieb_liniger_state::lieb_liniger_state(double new_c, double new_L,
                                       int new_N)
    : c(new_c),
      L(new_L),
      N(new_N),
      Is(N)
{
    std::cout << "Hello, this is the ground state." << std::endl;
    generate_gs_bethe_numbers();
    for (auto i: Is) {
        std::cout << i << std::endl;
    }
}

lieb_liniger_state::lieb_liniger_state(double new_c, double new_L,
                                       int new_N,
                                       std::vector<double> new_Is)
    : c(new_c),
      L(new_L),
      N(new_N),
      Is(new_Is)
{
    std::cout << "Hello, this is an excited state." << std::endl;
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
