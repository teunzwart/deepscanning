#include "lieb_liniger_state.hpp"

#include "catch.hpp"
#include <Eigen/Dense>

#include <iostream>

TEST_CASE("For N even, the ground state Bethe numbers and rapidities are correct") {
    lieb_liniger_state llstate(1, 100, 10);
    Eigen::VectorXd expected_Is(10);
    expected_Is << -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5;
    Eigen::VectorXd expected_rapidities(10);
    llstate.find_rapidities();
    expected_rapidities << -0.237119, -0.184234, -0.131491, -0.0788525, -0.0262771, 0.0262771, 0.0788525, 0.131491, 0.184234, 0.237119;
    REQUIRE(llstate.Is == expected_Is);
    REQUIRE(llstate.lambdas.isApprox(expected_rapidities, 10e-4) == true);

}

TEST_CASE("For N odd, the ground state Bethe numbers and rapidities are correct") {
    lieb_liniger_state llstate(1, 100, 9);
    Eigen::VectorXd expected_Is(9);
    expected_Is << -4, -3, -2, -1, 0, 1, 2, 3, 4;
    Eigen::VectorXd expected_rapidities(9);
    llstate.find_rapidities();
    expected_rapidities << -0.214027, -0.160379, -0.106851, -0.0534045, 0, 0.0534045, 0.106851, 0.160379, 0.214027;
    REQUIRE(llstate.Is == expected_Is);
    REQUIRE(llstate.lambdas.isApprox(expected_rapidities, 10e-4) == true);
}
