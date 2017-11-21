cmake_minimum_required(VERSION 3.9)

project(Bethe_Equation_Solver)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Weverything -Wno-padded -Wno-c++98-compat -ferror-limit=1 -MP -MMD")

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Protobuf REQUIRED)
find_package(TensorflowCC REQUIRED COMPONENTS Shared)


# execute_process(COMMAND llvm-config --includedir OUTPUT_VARIABLE OpenMP_CXX_FLAGS)
# # execute_process(COMMAND {llvm-config --includedir; echo omp.h} | xargs echo OUTPUT_VARIABLE OpenMP_EXE_LINKER_FLAGS)
# set(OpenMP_EXE_LINKER_FLAGS "/usr/local/Cellar/llvm/5.0.0/lib/clang/5.0.0/include/omp.h")
# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

find_package(Doxygen)
if (DOXYGEN_FOUND)
    # Set input and output files
    set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_BINARY_DIR}/Doxyfile)

    # Configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
else (DOXYGEN_FOUND)
     message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

enable_testing()

add_subdirectory("src")
add_subdirectory("test")
