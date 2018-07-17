/**
 * Calculate the displacement functions of the ground state for the Lieb-Liniger model.
 */

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

constexpr double PI = 3.141592653589793238463;

double kernel(const double k, const double c);
int theta(const double x);
double calculate_lf(const double lambda, const double lambda_prime, const double c, const double lambda_f, double precision);
double particle_displacement(double lambda, double lambda_p, double c, double lambda_f, double delta_x);
double hole_displacement(double lambda, double lambda_h, double c, double lambda_f, double delta_x);

/**
 * Calculate the Lieb-Liniger kernel
 */
double kernel(const double k, const double c) {
    return 2 * c / (k * k + c * c);
}

/**
 * Calculate the Heaviside theta function.
 */
inline int theta(const double x) {
    if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}

/**
 * Get the sign of a number.
 */
template <typename T>
int sign(const T x) {
    if (x >= 0) {
        return 1;
    } else {
        return -1;
    }
}


/**
 * Calculate the inverse of the truncated kernel L^(F) iteratively.
 */
double calculate_lf(const double lambda, const double lambda_prime, const double c, const double lambda_f, double precision=pow(10, -6)) {
    // Doing the truncation here in quicker than doing it everytime in the loop.
    if (theta(lambda_f - abs(lambda)) * theta(lambda_f - abs(lambda_prime)) == 0) {
        return 0;
    }
    double prev_lf = 1;
    double truncated_kernel =  kernel(lambda - lambda_prime, c);
    double integral = -2 * (atan((lambda_prime + lambda_f) / c) - atan((lambda_prime - lambda_f) / c));
    
    double lf = truncated_kernel;

    int it = 0;
    while (abs(lf - prev_lf) > precision) {
        prev_lf = lf;
        lf = truncated_kernel - lf * integral;
        // std::cout << "integral " << integral << std::endl;
        // std::cout << "trunc kernel " << truncated_kernel << std::endl;
        // std::cout << "it: " << it << ": " << lf << "\n";
        it += 1;
    }

    return lf;
}

double particle_displacement(double lambda, double lambda_p, double c, double lambda_f, double delta_x=0.01) {
    double non_integral_part = -1 / PI * sign(lambda_p) * theta(lambda_f - abs(lambda)) * atan(c / abs(lambda - lambda_p));
    int N = static_cast<int>(floor(2 * lambda_f / delta_x));
    double xn = -lambda_f;
    double integral = calculate_lf(lambda, xn, c, lambda_f) * atan(c / abs(xn - lambda_p));
    // We use the trapezoidal rule to calculate the integral.
    for (int k=1; k<N; k++) {
        xn += delta_x;
        integral += 2 * calculate_lf(lambda, xn, c, lambda_f) * atan(c / abs(xn - lambda_p));
    }
    xn += delta_x;
    integral += calculate_lf(lambda, xn, c, lambda_f) * atan(c / abs(xn - lambda_p));
    integral *= delta_x / 2;
    integral *= -1 / PI * sign(lambda_p);

    return non_integral_part + integral;
}


double hole_displacement(double lambda, double lambda_h, double c, double lambda_f, double delta_x=0.01) {
    double non_integral_part = 1 / PI * sign(lambda_h) * theta(lambda_f - abs(lambda)) * (atan((lambda - lambda_h) / c) - sign(lambda_h) * PI / 2);
    int N = static_cast<int>(floor(2 * lambda_f / delta_x));
    double xn = -lambda_f;
    double integral = calculate_lf(lambda, xn, c, lambda_f) * atan(c / abs(xn - lambda_h));
    // We use the trapezoidal rule to calculate the integral.
    for (int k=1; k<N; k++) {
        xn += delta_x;
        integral += 2 * calculate_lf(lambda, xn, c, lambda_f) *  (atan((lambda - lambda_h) / c) - sign(lambda_h) * PI / 2);
    }
    xn += delta_x;
    integral += calculate_lf(lambda, xn, c, lambda_f) *  (atan((lambda - lambda_h) / c) - sign(lambda_h) * PI / 2);
    integral *= delta_x / 2;
    integral *= 1 / PI;

    return non_integral_part + integral;
}

void create_lf_data() {
    double lambda_prime = 0;
    double lambda_f = 1;
    double c_iteration = 4;

    std::vector<double> lf(static_cast<std::vector<double>::size_type>(lambda_f * 2 / 0.01 + 1));
    for (std::vector<double>::size_type i = 0 ; i != lf.size(); i++) {
        lf[i] = particle_displacement(-lambda_f + 0.01 * i, lambda_prime, c_iteration, lambda_f) ;
    }
    for (auto lfi: lf) {
        std::cout << lfi << ", ";
    }

    std::cout << "\n\n\n" << std::endl;
}

void create_Dp_data() {
    double lambda_f = 1;
    double lambda_p = 1.5 ;
    double c_iteration = 4;

    std::vector<double> dps(static_cast<std::vector<double>::size_type>(lambda_f * 2 / 0.01 + 1));
    for (std::vector<double>::size_type i = 0 ; i != dps.size(); i++) {
        dps[i] = particle_displacement(-lambda_f + 0.01 * i, lambda_p, c_iteration, lambda_f) ;
    }
    for (auto dpi: dps) {
        std::cout << dpi << ", ";
    }
    std::cout << "\n\n\n" << std::endl;

}

void create_Dh_data() {
    double lambda_f = 1;
    double lambda_h = 0.5;
    double c_iteration = 4;

    std::vector<double> dps(static_cast<std::vector<double>::size_type>(lambda_f * 2 / 0.01 + 1));
    for (std::vector<double>::size_type i = 0 ; i != dps.size(); i++) {
        dps[i] = particle_displacement(-lambda_f + 0.01 * i, lambda_h, c_iteration, lambda_f);
    }
    for (auto dpi: dps) {
        std::cout << dpi << ", ";
    }
    std::cout << "\n\n\n" << std::endl;

}

int main() {
    // create_lf_data();
    // create_Dp_data();
    create_Dh_data();
}
