include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_BINARY_DIR})

add_library(machine_learning machine_learning.cpp)

add_library(lieb_liniger_state lieb_liniger_state.cpp)
target_link_libraries(lieb_liniger_state machine_learning Eigen3::Eigen)

add_executable(main main.cpp)
target_link_libraries(main lieb_liniger_state Eigen3::Eigen)
