# Bethe Equation Solver

A proof of concept for solving Bethe Ansatz equations for the Lieb-Liniger model using machine learning.

## Building
We use `CMake` as a build system. `Eigen` is used for linear algebra operations and 'Doxygen' for documentation.
To build:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug  # Or Release
make
```
