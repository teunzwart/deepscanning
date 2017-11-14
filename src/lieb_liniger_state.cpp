#include "lieb_liniger_state.hpp"

#include <iostream>
#include <vector>


lieb_liniger_state::lieb_liniger_state(double new_c, double new_L, double new_N)
    : c(new_c),
      L(new_L),
      N(new_N)
{
    std::cout << "Hello" << std::endl;
}


int main() {
    lieb_liniger_state test(1, 100, 100);
}
