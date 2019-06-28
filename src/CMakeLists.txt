# Let us define a library with the listed cpp files.
add_library(linalg
coomatrix.cpp
densematrix.cpp
operator.cpp
sparsematrix.cpp
vector.cpp
vectorview.cpp
)

# Let our "main.cpp" know where the library is.
target_include_directories(linalg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})