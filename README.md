# Bethe Equation Solver

A proof of concept for solving Bethe Ansatz equations for the Lieb-Liniger model using machine learning.

## Building
Compilation is done with `clang++`. We use `CMake` as a build system. `Eigen` is used for linear algebra operations and `Doxygen` for documentation.
To install `Eigen` and `Doxygen` with `Homebrew`, run

``` bash
brew install eigen doxygen
```
`Tensorflow` is used for machine learning. To build a version compatible with `CMake` (and to not be required to have all code in the `Tensorflow` source tree), we build a shared library using the instructions [here](https://github.com/FloopCZ/tensorflow_cc).

To build:
```bash
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug  # Or Release
make
```
