cmake_minimum_required(VERSION 3.9)

project(Bethe_Equation_Solver)

set(CMAKE_CXX, "clang++")
set(CMAKE_CXX_FLAGS "-std=c++14 -Weverything -ferror-limit=1 -MP -MMD")

add_subdirectory("src")

