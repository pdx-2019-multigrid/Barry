# Let us define a library with the listed cpp files.
add_library(linalg
vectorview.cpp
vector.cpp
operator.cpp
densematrix.cpp
sparsematrix.cpp
coomatrix.cpp
)

# Let our "main.cpp" know where the library is.
target_include_directories(linalg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
