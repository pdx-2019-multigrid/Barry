cmake_minimum_required(VERSION 3.14)

# Use C++11
set(CMAKE_CXX_STANDARD 11)

# We have no C files and can tell CMake that.
project(LANGUAGES CXX) 

# Go to CMake file in directory src. Use the source
# files there, a library called linalg.
add_subdirectory(src)

# Make an executable files.
add_executable(test pcg.cpp)


# Link library linalg into executable test.
target_link_libraries(test PRIVATE linalg)

# Copy Coo list files into build directory.
function(copy filename)
  configure_file(
    "${PROJECT_SOURCE_DIR}/data/${filename}"
    "${PROJECT_BINARY_DIR}/${filename}" COPYONLY)
endfunction(copy)

copy(34.txt)
copy(665.txt)
copy(6k.txt)