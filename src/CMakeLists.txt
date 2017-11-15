include_directories(${Bethe_Equation_Solver_SOURCE_DIR}/include)
link_directories(${Bethe_Equation_Solver_BINARY_DIR})

add_library(lieb_liniger_state lieb_liniger_state.cpp)
target_link_libraries(lieb_liniger_state Eigen3::Eigen)

add_executable(main main.cpp)
target_link_libraries(main lieb_liniger_state Eigen3::Eigen)
