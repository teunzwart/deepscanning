/**
 * Define global constants.
 */

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>
#include <limits>

const double PI = 3.141592653589793238462643;

const double MACHINE_EPS = std::numeric_limits<double>::epsilon();
const double MACHINE_EPS_SQUARE = std::pow(MACHINE_EPS, 2.0);

#endif // CONSTANTS_H
