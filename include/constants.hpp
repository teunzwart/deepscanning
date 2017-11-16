/**
 * Define global constants.
 */

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>
#include <limits>

const double PI = 3.141592653589793238462643;

constexpr double MACHINE_EPS = std::numeric_limits<double>::epsilon();
constexpr double MACHINE_EPS_SQUARE = MACHINE_EPS * MACHINE_EPS;

#endif // CONSTANTS_H
