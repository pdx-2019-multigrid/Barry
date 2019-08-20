/*! @file */

#ifndef PARTITION_HPP__
#define PARTITION_HPP__

#include "metis.h"
#include "vectorview.hpp"
#include "parser.hpp"
#include <vector>
#include <cmath>

namespace linalgcpp
{

template<typename T>
Vector<int> Partition(const SparseMatrix<T> matrix, int nparts)
{    
    int error;
    int objval;
    int * ncon = new int(1); // The number of balancing constraints.
    
    int options[METIS_NOPTIONS] = {};

    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_NUMBERING] = 0; // Start numbering parts in partion from zero.
    // options[METIS_OPTION_CONTIG] = 1; // 1 Force continguous partition.
    
    int * nvtxs = new int(matrix.Cols()); // The number of vertices in graph.
    std::vector<int> part(* nvtxs);
    
    // std::cout << "nvtxs = " << * nvtxs << std::endl;
    // std::cout << "Size of part = " << part.size() << std::endl;

    std::vector<int> xadj(matrix.GetIndptr());
    std::vector<int> adjncy(matrix.GetIndices());
    
    error = METIS_PartGraphKway(nvtxs,
                                ncon,
                                xadj.data(),
                                adjncy.data(),
                                NULL,
                                NULL,
                                NULL,
                                &nparts,
                                NULL,
                                NULL,
                                options,
                                &objval, // Why cannot this be a pointer?
                                part.data()
                                );
    
    Vector<int> groups(part);
    delete ncon;

    return groups;
}

// This assumes there are no empty partitions / skipped integers
SparseMatrix<double> Weighted(const Vector<int> partitions)
{
    int rows = partitions.size();
    int cols = Max(partitions) + 1;
    
    std::vector<int> indptr(rows + 1);
    std::vector<double> data(rows);
    
    std::vector<int> counts(cols, 0);
    for(int partition : partitions) ++counts[partition];
    for(size_t node = 0; node < rows; ++node)
    {
        data[node] = 1.0 / std::sqrt(counts[partitions[node]]);
        indptr[node] = node;
    }
    indptr[rows] = rows;
    
    SparseMatrix<double> interpolation_matrix(indptr, partitions.data(), data, rows, cols);
    return interpolation_matrix;
}

SparseMatrix<int> Unweighted(const Vector<int> partitions)
{
    int rows = partitions.size();
    int cols = Max(partitions) + 1;
    
    std::vector<int> indptr(rows + 1);
    std::vector<int> data(rows, 1);
    
    for(size_t node = 0; node < rows; ++node)
    {
        indptr[node] = node;
    }
    indptr[rows] = rows;
    
    SparseMatrix<int> interpolation_matrix(indptr, partitions.data(), data, rows, cols);
    return interpolation_matrix;
}

} // namespace linalgcpp

#endif // PARTITION_HPP__
