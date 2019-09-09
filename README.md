* Both the source files and header files are in [`src`](src).
* The directory [`data`](data) contains sample matrices and vectors.
* Rough two-level and multilevel preconditioners. Each has own PCG solver, which is perhaps not ideal.
* Slightly altered Caitlin's partition.hpp file.
* Note: `METIS` does not like `nparts = 1`.
* Use the standard CMake procedure: `mkdir build && cd build && cmake ..`
* ~~Todo: allow for different smoothers.~~