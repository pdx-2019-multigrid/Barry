cmake_minimum_required(VERSION 3.5)

# Use C++11
set(CMAKE_CXX_STANDARD 11)

# Use gcc-9.1.0
set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-9")

# Use OpenMP
# set(CMAKE_CXX_FLAGS "-fopenmp")

# Do not use debug mode.
set(CMAKE_BUILD_TYPE Release)

# We have no C files and can tell CMake that.
project(LANGUAGES CXX)

# The include directory contains metis.h
# and the lib directory contains libmetis.a or libmetis.so
include_directories("/usr/local/include")
link_directories("/usr/local/lib")

# Go to CMake file in directory src. Use the
# source files there, a library called linalg.
add_subdirectory(src)

# Make an executable file.
add_executable(pcg pcg.cpp)

# Link library linalg into executable file.
target_link_libraries(pcg PRIVATE linalg metis)
