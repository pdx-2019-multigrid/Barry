/*! @file */
#ifndef PARSER_HPP__
#define PARSER_HPP__

#include <memory>
#include <vector>
#include <assert.h>
#include <fstream>

#include "vector.hpp"
#include "sparsematrix.hpp"
#include "coomatrix.hpp"

namespace linalgcpp
{

/*! @brief Read a text file from disk.
    @param file_name file to read
    @retval vector of data read from disk
*/
template <typename T = double>
std::vector<T> ReadText(const std::string& file_name)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    std::vector<T> data;

    for (T val; file >> val; )
    {
        data.push_back(val);
    }

    file.close();

    return data;
}

/*! @brief Write a vector to a text file on disk.
    @param vect vector to write
    @param file_name file to write to
*/
template <typename T>
void WriteText(const T& vect, const std::string& file_name)
{
    std::ofstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    file.precision(16);

    for (const auto& val : vect)
    {
        file << val << "\n";
    }
}

/*! @brief Write a vector to a text file on disk in binary format
     Format: size data data data ...

    @param vect vector to write
    @param file_name file to write to
*/
template <typename T = double>
void WriteBinary(const std::vector<T>& vect, const std::string& file_name)
{
    std::ofstream file(file_name.c_str(), std::ios::out | std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    size_t size = vect.size();

    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(reinterpret_cast<const char*>(&vect[0]), size * sizeof(T));

    file.close();
}

/*! @brief Read a vector to a text file on disk in binary format
     Format: size data data data ...

    @param vect vector to write
    @param file_name file to write to
*/
template <typename T = double>
std::vector<T> ReadBinaryVect(const std::string& file_name)
{
    std::ifstream file(file_name.c_str(), std::ios::binary | std::ios::in);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    size_t size = 0;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));

    std::vector<T> vect(size);

    file.read(reinterpret_cast<char*>(&vect[0]), size * sizeof(T));
    file.close();

    return vect;
}


/*! @brief Write a csr matrix to text file on disk in binary format
     Format:
     rows cols nnz indptr indices data

    @param vect vector to write
    @param file_name file to write to
*/
template <typename T = double>
void WriteBinary(const SparseMatrix<T>& mat, const std::string& file_name)
{
    std::ofstream file(file_name.c_str(), std::ios::out | std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    size_t rows = mat.Rows();
    size_t cols = mat.Cols();
    size_t nnz = mat.nnz();

    const auto& indptr = mat.GetIndptr();
    const auto& indices = mat.GetIndices();
    const auto& data = mat.GetData();

    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));

    file.write(reinterpret_cast<const char*>(&indptr[0]), (rows + 1) * sizeof(int));
    file.write(reinterpret_cast<const char*>(&indices[0]), nnz * sizeof(int));
    file.write(reinterpret_cast<const char*>(&data[0]), nnz * sizeof(T));

    file.close();
}

/*! @brief Read a csr matrix from a text file on disk in binary format
     Format:
     rows cols nnz indptr indices data

    @param vect vector to write
    @param file_name file to write to
*/
template <typename T = double>
SparseMatrix<T> ReadBinaryMat(const std::string& file_name)
{
    std::ifstream file(file_name.c_str(), std::ios::out | std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    size_t rows;
    size_t cols;
    size_t nnz;

    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    file.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));

    std::vector<int> indptr(rows + 1);
    std::vector<int> indices(nnz);
    std::vector<T> data(nnz);

    file.read(reinterpret_cast<char*>(&indptr[0]), (rows + 1) * sizeof(int));
    file.read(reinterpret_cast<char*>(&indices[0]), nnz * sizeof(int));
    file.read(reinterpret_cast<char*>(&data[0]), nnz * sizeof(T));

    file.close();

    return SparseMatrix<T>(std::move(indptr), std::move(indices), std::move(data), rows, cols);
}

/*! @brief Read an adjacency list from disk.
    Data is expected to be formatted as :
       i j
       i j
       i j
       ...
    @param file_name file to read
    @param symmetric if true the file only contain values above
    or below the diagonal and the diagonal itself. The other corresponding
    symmetric values will be added to the matrix.
*/
template <typename T = double>
SparseMatrix<T> ReadAdjList(const std::string& file_name, bool symmetric = false)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    CooMatrix<T> coo;

    int i;
    int j;
    T val = 1;

    while (file >> i >> j)
    {
        coo.Add(i, j, val);

        if (symmetric && i != j)
        {
            coo.Add(j, i, val);
        }
    }

    file.close();

    return coo.ToSparse();
}

/*! @brief Read graph laplacian from an adjacency list from disk.
    Data is expected to be formatted as :
       i j
       i j
       i j
       ...
    @param file_name file to read
*/
template <typename T = double>
SparseMatrix<T> ReadGraphList(const std::string& file_name)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    CooMatrix<T> coo;

    int i;
    int j;

    while (file >> i >> j)
    {
        if (i < j)
        {
            coo.Add(j, i, -1);
            coo.Add(i, j, -1);
            coo.Add(i, i, 1);
            coo.Add(j, j, 1);
        }
    }

    file.close();

    return coo.ToSparse();
}

/*! @brief Write an adjacency list to disk.
    @param mat matrix to write out
    @param file_name file to write to
    @param symmetric if true only write entries above and including the diagonal.
    Otherwise write out all entries

    @note see ReadAdjList for format description
*/
template <typename T = double>
void WriteAdjList(const SparseMatrix<T>& mat, const std::string& file_name, bool symmetric = false)
{
    std::ofstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    const std::vector<int>& indptr = mat.GetIndptr();
    const std::vector<int>& indices = mat.GetIndices();

    const int rows = mat.Rows();

    for (int i = 0; i < rows; ++i)
    {
        for (int j = indptr[i]; j < indptr[i + 1]; ++j)
        {
            const int col = indices[j];

            if (!symmetric || col >= i)
            {
                file << i << " " << col << "\n";
            }
        }
    }

    file.close();
}

/*! @brief Read a coordinate list from disk.
    Data is expected to be formatted as :
       i j val
       i j val
       i j val
       ...
    @param file_name file to read
    @param symmetric if true the file only contain values above
    or below the diagonal and the diagonal itself. The other corresponding
    symmetric values will be added to the matrix.
*/
template <typename T = double>
SparseMatrix<T> ReadCooList(const std::string& file_name, bool symmetric = false)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    CooMatrix<T> coo;

    int i;
    int j;
    T val;

    while (file >> i >> j >> val)
    {
        coo.Add(i, j, val);

        if (symmetric && i != j)
        {
            coo.Add(j, i, val);
        }
    }

    file.close();

    return coo.ToSparse();
}

/*! @brief Write a coordinate list to disk.
    @param mat matrix to write out
    @param file_name file to write to
    @param symmetric if true only write entries above and including the diagonal.
    Otherwise write out all entries

    @note see ReadCooList for format description
*/
template <typename T = double>
void WriteCooList(const SparseMatrix<T>& mat, const std::string& file_name, bool symmetric = false)
{
    std::ofstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    const std::vector<int>& indptr = mat.GetIndptr();
    const std::vector<int>& indices = mat.GetIndices();
    const std::vector<double>& data = mat.GetData();

    const int rows = mat.Rows();

    for (int i = 0; i < rows; ++i)
    {
        for (int j = indptr[i]; j < indptr[i + 1]; ++j)
        {
            const int col = indices[j];

            if (!symmetric || col >= i)
            {
                file << i << " " << col << " " << data[j] << "\n";
            }
        }
    }

    file.close();
}

/*! @brief Read a table from file from disk.
    Data is expected to be formatted as :
        0 1 2 3 4
        5 8 0
        1 2 3 10
    Where each row corresponds to the entries in that row
    @param file_name file to read
    @retval SparseMatrix matrix of data read from disk
*/
template <typename T = int>
SparseMatrix<T> ReadTable(const std::string& file_name)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    std::vector<int> indptr(1, 0);
    std::vector<int> indices;

    for (std::string line; std::getline(file, line); )
    {
        std::stringstream stream(line);

        for (int index; stream >> index; )
        {
            indices.push_back(index);
        }

        indptr.push_back(indices.size());
    }

    file.close();

    const int rows = indptr.size() - 1;
    const int cols = *std::max_element(begin(indices), end(indices)) + 1;

    std::vector<T> data(indices.size(), 1);

    return SparseMatrix<T>(std::move(indptr), std::move(indices), std::move(data),
                           rows, cols);
}

/*! @brief Write a table to a file on disk.
    @param mat table to write
    @param file_name file to write to

    @note see ReadTable for format description
*/
template <typename T = int>
void WriteTable(const SparseMatrix<T>& mat, const std::string& file_name)
{
    std::ofstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    const std::vector<int>& indptr = mat.GetIndptr();
    const std::vector<int>& indices = mat.GetIndices();

    const int rows = mat.Rows();

    for (int i = 0; i < rows; ++i)
    {
        for (int j = indptr[i]; j < indptr[i + 1]; ++j)
        {
            const std::string space = j + 1 == indptr[i + 1] ? "" : " ";
            file << indices[j] << space;
        }

        file << "\n";
    }

    file.close();
}

/*! @brief Read a CSR Matrix from disk.
    rows cols indptr indices data

    @param file_name file to read
    @retval SparseMatrix matrix of data read from disk
*/
template <typename T = double>
SparseMatrix<T> ReadCSR(const std::string& file_name)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    size_t rows;
    size_t cols;

    file >> rows >> cols;

    std::vector<int> indptr(rows + 1);

    for (size_t i = 0; i < rows + 1; ++i)
    {
        file >> indptr[i];
    }

    int nnz = indptr.back();

    std::vector<int> indices(nnz);
    std::vector<T> data(nnz);

    for (int i = 0; i < nnz; ++i)
    {
        file >> indices[i];
    }

    for (int i = 0; i < nnz; ++i)
    {
        file >> data[i];
    }

    file.close();

    return SparseMatrix<T>(std::move(indptr), std::move(indices), std::move(data),
                           rows, cols);
}

/*! @brief Write a CSR Matrix to disk.
    rows cols indptr indices data

    @param SparseMatrix matrix of data read from disk
    @param file_name file to read
*/
template <typename T = double>
void WriteCSR(const SparseMatrix<T>& mat, const std::string& file_name)
{
    std::ofstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    const std::vector<int>& indptr = mat.GetIndptr();
    const std::vector<int>& indices = mat.GetIndices();
    const std::vector<T>& data = mat.GetData();

    const int rows = mat.Rows();
    const int cols = mat.Cols();

    file << rows << "\n" << cols << "\n";

    for (auto i : indptr)
    {
        file << i << "\n";
    }

    for (auto i : indices)
    {
        file << i << "\n";
    }

    file.precision(16);

    for (auto i : data)
    {
        file << i << "\n";
    }

    file.close();
}

/*! @brief Read matrix market coordinate format from disk.
    @warning input file is 1 based
    Data is expected to be formatted as:
       rows cols nnz
       i j val
       i j val
       i j val
       ...
    @param file_name file to read
*/
template <typename T = double>
SparseMatrix<T> ReadMTX(const std::string& file_name)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    bool symmetric = false;
    std::string line;

    while (getline(file, line) && line.size() && line[0] == '%')
    {
        symmetric |= (line.find("symmetric") != std::string::npos);
    }

    int rows;
    int cols;
    int nnz;

    // First line after comments contains size info
    std::istringstream(line) >> rows >> cols >> nnz;

    CooMatrix<T> coo(rows, cols);
    coo.Reserve(nnz);

    int base = 1;
    int i;
    int j;
    T val;

    while (file >> i >> j >> val)
    {
        coo.Add(i - base, j - base, val);

        if (symmetric && i != j)
        {
            coo.Add(j - base, i - base, val);
        }
    }

    file.close();

    return coo.ToSparse();
}

/*! @brief Read matrix market coordinate format from disk.
    @warning input file is 1 based
    Data is expected to be formatted as:
       rows cols nnz
       i j
       i j
       i j
       ...
    @param file_name file to read
*/
template <typename T = double>
SparseMatrix<T> ReadMTXList(const std::string& file_name)
{
    std::ifstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    bool symmetric = false;
    std::string line;

    while (getline(file, line) && line.size() && line[0] == '%')
    {
        symmetric |= (line.find("symmetric") != std::string::npos);
    }

    int rows;
    int cols;
    int nnz;

    // First line after comments contains size info
    std::istringstream(line) >> rows >> cols >> nnz;

    CooMatrix<T> coo(rows, cols);
    coo.Reserve(nnz);

    int base = 1;
    int i;
    int j;
    T val = 1.0;

    while (file >> i >> j)
    {
        coo.Add(i - base, j - base, val);

        if (symmetric && i != j)
        {
            coo.Add(j - base, i - base, val);
        }
    }

    file.close();

    return coo.ToSparse();
}

/*! @brief Write matrix market coordinate format to disk.
    @warning output file is 1 based
    Data is expected to be formatted as:
       rows cols nnz
       i j val
       i j val
       i j val
       ...
    @param mat matrix to save
    @param file_name file to write to
*/
template <typename T = double>
void WriteMTX(const SparseMatrix<T>& mat, const std::string& file_name)
{
    std::ofstream file(file_name.c_str());

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    int base = 1;

    int rows = mat.Rows();
    int cols = mat.Cols();
    int nnz = mat.nnz();

    file.precision(16);
    file << "%MatrixMarket matrix coordinate real general\n";
    file << rows << " " << cols << " " << nnz <<  "\n";

    const std::vector<int>& indptr = mat.GetIndptr();
    const std::vector<int>& indices = mat.GetIndices();
    const std::vector<T>& data = mat.GetData();

    for (int i = 0; i < rows; ++i)
    {
        for (int j = indptr[i]; j < indptr[i + 1]; ++j)
        {
            file << i + base << " " << indices[j] + 1 << " " << data[j] << "\n";
        }
    }

    file.close();
}

} // namespace mylinalg

#endif // PARSER_HPP__
