# Configure Catch testing framework.
set(CATCH_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/external/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/include)
add_library(test_main test_main.cpp)
target_link_libraries(test_main Catch)

add_executable(tests test_lieb_liniger_state.cpp)
target_link_libraries(tests lieb_liniger_state test_main Catch)

add_test(NAME tests COMMAND tests)

