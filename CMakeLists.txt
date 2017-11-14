cmake_minimum_required(VERSION 3.9)

project(Bethe_Equation_Solver)

set(CMAKE_CXX, "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Weverything -Wno-padded -Wno-c++98-compat -ferror-limit=1 -MP -MMD")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Bethe_Equation_Solver_SOURCE_DIR}/bin)

add_subdirectory("src")

