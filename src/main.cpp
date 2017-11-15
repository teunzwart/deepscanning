#include "lieb_liniger_state.hpp"

int main() {
    lieb_liniger_state ll_state(1, 10, 20);
    ll_state.calculate_rapidities();
}
