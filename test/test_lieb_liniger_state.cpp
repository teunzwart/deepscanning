#include "lieb_liniger_state.hpp"

#include "catch.hpp"

#include <iostream>

TEST_CASE("Lieb-Liniger states can be created and worked on.") {
    lieb_liniger_state llstate(1, 100, 100);
}
