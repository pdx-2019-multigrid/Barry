/*! @file */

#ifndef COOMATRIX_HPP__
#define COOMATRIX_HPP__

#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <tuple>
#include <assert.h>

#include <unordered_map>

#include "sparsematrix.hpp"
#include "densematrix.hpp"

namespace linalgcpp
{

/*! @brief Coordinate Matrix that keeps track
           of individual entries in a matrix.

    @note Multiple entries for a single coordinate
    are summed together
*/
template <typename T>
class CooMatrix : public Operator
{
    public:
        /*! @brief Default Constructor

            @note If the matrix size was not specified during
            creation, then the number of rows is determined
            by the maximum element.
        */
        CooMatrix();

        /*! @brief Square Constructor
            @param size the number of rows and columns
        */
        explicit CooMatrix(int size);

        /*! @brief Rectangle Constructor
            @param rows the number of rows
            @param cols the number of columns
        */
        CooMatrix(int rows, int cols);

        /*! @brief Copy Constructor */
        CooMatrix(const CooMatrix& other) noexcept;

        /*! @brief Move Constructor */
        CooMatrix(CooMatrix&& other) noexcept;

        /*! @brief Assignment Operator */
        CooMatrix& operator=(CooMatrix other) noexcept;

        /*! @brief Destructor */
        ~CooMatrix() noexcept = default;

        /*! @brief Swap two matrices
            @param lhs left hand side matrix
            @param rhs right hand side matrix
        */
        template <typename U>
        friend void swap(CooMatrix<U>& lhs, CooMatrix<U>& rhs) noexcept;

        /*! @brief Set the size of the matrix
            @param rows the number of rows
            @param cols the number of columns
        */
        void SetSize(int rows, int cols);

        /*! @brief Reserve space for entries
            @param size the number of entries
        */
        void Reserve(int size);

        /*! @brief Add an entry to the matrix
            @param i row index
            @param j column index
            @param val value to add
        */
        void Add(int i, int j, T val);

        /*! @brief Add an entry to the matrix and its symmetric counterpart
            @param i row index
            @param j column index
            @param val value to add
        */
        void AddSym(int i, int j, T val);

        /*! @brief Add a dense matrix worth of entries
            @param indices row and column indices to add
            @param values the values to add
        */

        void Add(const std::vector<int>& indices,
                 const DenseMatrix& values);

        /*! @brief Add a dense matrix worth of entries
            @param rows set of row indices
            @param cols set of column indices
            @param values the values to add
        */
        void Add(const std::vector<int>& rows,
                 const std::vector<int>& cols,
                 const DenseMatrix& values);

        /*! @brief Add a scaled dense matrix worth of entries
            @param indices row and column indices to add
            @param scale scale to apply to added values
            @param values the values to add
        */

        void Add(const std::vector<int>& indices, T scale,
                 const DenseMatrix& values);

        /*! @brief Add a dense matrix worth of entries
            @param rows set of row indices
            @param cols set of column indices
            @param scale scale to apply to added values
            @param values the values to add
        */
        void Add(const std::vector<int>& rows,
                 const std::vector<int>& cols,
                 T scale,
                 const DenseMatrix& values);

        /*! @brief Add a vector worth of entries
            @param rows set of row indices
            @param col col index
            @param values the values to add
        */
        void Add(const std::vector<int>& rows, int col,
                 const VectorView<T>& values);

        /*! @brief Add a transpose vector worth of entries
            @param row row index
            @param cols set of column indices
            @param values the values to add
        */
        void Add(int row, const std::vector<int>& cols,
                 const VectorView<T>& values);

        /*! @brief Add a scaled vector worth of entries
            @param rows set of row indices
            @param col col index
            @param scale scale to apply to added values
            @param values the values to add
        */
        void Add(const std::vector<int>& rows, int col, T scale,
                 const VectorView<T>& values);

        /*! @brief Add a transpose vector worth of entries
            @param row row index
            @param cols set of column indices
            @param scale scale to apply to added values
            @param values the values to add
        */
        void Add(int row, const std::vector<int>& cols, T scale,
                 const VectorView<T>& values);

        /*! @brief Add an indexable object worth of entries
            @param rows set of row indices
            @param cols set of column indices
            @param values the values to add
        */
        template <typename U>
        void Add(const std::vector<int>& rows, const std::vector<int>& cols,
                 const U& values);

        /*! @brief Add an indexable object worth of entries
            @param rows set of row indices
            @param cols set of column indices
            @param scale scale to apply to added values
            @param values the values to add
        */
        template <typename U>
        void Add(const std::vector<int>& rows, const std::vector<int>& cols, T scale,
                 const U& values);

        /*! @brief Permute the rows
            @param perm permutation to apply
        */
        void PermuteRows(const std::vector<int>& perm);

        /*! @brief Permute the columns
            @param perm permutation to apply
        */
        void PermuteCols(const std::vector<int>& perm);

        /*! @brief Permute both rows and columns
            @param row_perm permutation to apply to rows
            @param col_perm permutation to apply to cols
        */
        void PermuteRowsCols(const std::vector<int>& row_perm, const std::vector<int>& col_perm);

        /*! @brief Generate a sparse matrix from the entries
            @retval SparseMatrix containing all the entries

            @note Multiple entries for a single coordinate
            are summed together
        */
        template <typename U = T>
        SparseMatrix<U> ToSparse() const;

        /*! @brief Generate a dense matrix from the entries
            @retval DenseMatrix containing all the entries

            @note Multiple entries for a single coordinate
            are summed together
        */
        DenseMatrix ToDense() const;

        /*! @brief Generate a dense matrix from the entries
            @param DenseMatrix containing all the entries

            @note Multiple entries for a single coordinate
            are summed together
        */
        void ToDense(DenseMatrix& dense) const;

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @param output the output vector y
        */
        void Mult(const VectorView<double>& input, VectorView<double> output) const override;

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @param output the output vector y
        */
        void MultAT(const VectorView<double>& input, VectorView<double> output) const override;

        /*! @brief Print all entries
            @param label label to print before data
            @param out stream to print to
        */
        void Print(const std::string& label = "", std::ostream& out = std::cout) const;

        /*! @brief Eliminate entries with zero value
            @param tolerance how small of values to erase
        */
        void EliminateZeros(double tolerance = 0);

    private:
        std::tuple<int, int> FindSize() const;

        bool size_set_;

        mutable std::vector<std::tuple<int, int, T>> entries_;
};

template <typename T>
CooMatrix<T>::CooMatrix()
    : size_set_(false)
{
}

template <typename T>
CooMatrix<T>::CooMatrix(int size) : CooMatrix(size, size)
{

}

template <typename T>
CooMatrix<T>::CooMatrix(int rows, int cols)
    : Operator(rows, cols), size_set_(true)
{
}

template <typename T>
CooMatrix<T>::CooMatrix(const CooMatrix& other) noexcept
    : Operator(other), size_set_(other.size_set_), entries_(other.entries_)
{

}

template <typename T>
CooMatrix<T>::CooMatrix(CooMatrix&& other) noexcept
{
    swap(*this, other);
}

template <typename T>
CooMatrix<T>& CooMatrix<T>::operator=(CooMatrix other) noexcept
{
    swap(*this, other);

    return *this;
}

template <typename T>
void swap(CooMatrix<T>& lhs, CooMatrix<T>& rhs) noexcept
{
    swap(static_cast<Operator&>(lhs), static_cast<Operator&>(rhs));

    std::swap(lhs.size_set_, rhs.size_set_);
    std::swap(lhs.entries_, rhs.entries_);
}

template <typename T>
void CooMatrix<T>::SetSize(int rows, int cols)
{
    assert(rows >= 0);
    assert(cols >= 0);

    size_set_ = true;

    rows_ = rows;
    cols_ = cols;
}

template <typename T>
void CooMatrix<T>::Reserve(int size)
{
    assert(size >= 0);

    entries_.reserve(size);
}

template <typename T>
void CooMatrix<T>::Add(int i, int j, T val)
{
    assert(i >= 0);
    assert(j >= 0);
    assert(val == val); // is finite

    if (size_set_)
    {
        assert(i < rows_);
        assert(j < cols_);
    }

    entries_.emplace_back(i, j, val);
}

template <typename T>
void CooMatrix<T>::AddSym(int i, int j, T val)
{
    Add(i, j, val);

    if (i != j)
    {
        Add(j, i, val);
    }
}

template <typename T>
void CooMatrix<T>::Add(const std::vector<int>& indices,
                       const DenseMatrix& values)
{
    Add(indices, indices, values);
}

template <typename T>
void CooMatrix<T>::Add(const std::vector<int>& rows,
                       const std::vector<int>& cols,
                       const DenseMatrix& values)
{
    assert(rows.size() == static_cast<unsigned int>(values.Rows()));
    assert(cols.size() == static_cast<unsigned int>(values.Cols()));

    const int num_rows = values.Rows();
    const int num_cols = values.Cols();

    for (int j = 0; j < num_cols; ++j)
    {
        const int col = cols[j];

        for (int i = 0; i < num_rows; ++i)
        {
            const int row = rows[i];
            const double val = values(i, j);

            Add(row, col, val);
        }
    }
}

template <typename T>
void CooMatrix<T>::Add(const std::vector<int>& indices, T scale,
                       const DenseMatrix& values)
{
    Add(indices, indices, scale, values);
}

template <typename T>
void CooMatrix<T>::Add(const std::vector<int>& rows,
                       const std::vector<int>& cols,
                       T scale,
                       const DenseMatrix& values)
{
    assert(rows.size() == static_cast<unsigned int>(values.Rows()));
    assert(cols.size() == static_cast<unsigned int>(values.Cols()));

    const int num_rows = values.Rows();
    const int num_cols = values.Cols();

    for (int j = 0; j < num_cols; ++j)
    {
        const int col = cols[j];

        for (int i = 0; i < num_rows; ++i)
        {
            const int row = rows[i];
            const double val = values(i, j);

            Add(row, col, scale * val);
        }
    }
}


template <typename T>
void CooMatrix<T>::Add(const std::vector<int>& rows, int col,
                       const VectorView<T>& values)
{
    assert(rows.size() == static_cast<unsigned int>(values.size()));

    int size = rows.size();

    for (int i = 0; i < size; ++i)
    {
        Add(rows[i], col, values[i]);
    }
}

template <typename T>
void CooMatrix<T>::Add(int row, const std::vector<int>& cols,
                       const VectorView<T>& values)
{
    assert(cols.size() == static_cast<unsigned int>(values.size()));

    int size = cols.size();

    for (int i = 0; i < size; ++i)
    {
        Add(row, cols[i], values[i]);
    }
}

template <typename T>
void CooMatrix<T>::Add(const std::vector<int>& rows, int col, T scale,
                       const VectorView<T>& values)
{
    assert(rows.size() == static_cast<unsigned int>(values.size()));

    int size = rows.size();

    for (int i = 0; i < size; ++i)
    {
        Add(rows[i], col, scale * values[i]);
    }
}

template <typename T>
void CooMatrix<T>::Add(int row, const std::vector<int>& cols, T scale,
                       const VectorView<T>& values)
{
    assert(cols.size() == static_cast<unsigned int>(values.size()));

    int size = cols.size();

    for (int i = 0; i < size; ++i)
    {
        Add(row, cols[i], scale * values[i]);
    }
}

template <typename T>
template <typename U>
void CooMatrix<T>::Add(const std::vector<int>& rows, const std::vector<int>& cols,
                       const U& values)
{
    assert(rows.size() == static_cast<unsigned int>(values.size()));
    assert(cols.size() == static_cast<unsigned int>(values.size()));

    int size = cols.size();

    for (int i = 0; i < size; ++i)
    {
        Add(rows[i], cols[i], values[i]);
    }
}

template <typename T>
template <typename U>
void CooMatrix<T>::Add(const std::vector<int>& rows, const std::vector<int>& cols, T scale,
                       const U& values)
{
    assert(rows.size() == static_cast<unsigned int>(values.size()));
    assert(cols.size() == static_cast<unsigned int>(values.size()));

    int size = cols.size();

    for (int i = 0; i < size; ++i)
    {
        Add(rows[i], cols[i], scale * values[i]);
    }
}

template <typename T>
DenseMatrix CooMatrix<T>::ToDense() const
{
    DenseMatrix dense;
    ToDense(dense);

    return dense;
}

template <typename T>
void CooMatrix<T>::ToDense(DenseMatrix& dense) const
{

    int rows;
    int cols;
    std::tie(rows, cols) = FindSize();

    dense.SetSize(rows, cols);
    dense = 0.0;

    for (const auto& entry : entries_)
    {
        int i = std::get<0>(entry);
        int j = std::get<1>(entry);
        double val = std::get<2>(entry);

        dense(i, j) += val;
    }
}

template <typename T>
void CooMatrix<T>::PermuteRows(const std::vector<int>& perm)
{
    int rows;
    int cols;
    std::tie(rows, cols) = FindSize();

    assert(static_cast<int>(perm.size()) == rows);

    for (auto& entry : entries_)
    {
        std::get<0>(entry) = perm[std::get<0>(entry)];
    }
}

template <typename T>
void CooMatrix<T>::PermuteCols(const std::vector<int>& perm)
{
    int rows;
    int cols;
    std::tie(rows, cols) = FindSize();

    assert(static_cast<int>(perm.size()) == cols);

    for (auto& entry : entries_)
    {
        std::get<1>(entry) = perm[std::get<1>(entry)];
    }
}

template <typename T>
void CooMatrix<T>::PermuteRowsCols(const std::vector<int>& row_perm, const std::vector<int>& col_perm)
{
    int rows;
    int cols;
    std::tie(rows, cols) = FindSize();

    assert(static_cast<int>(row_perm.size()) == rows);
    assert(static_cast<int>(col_perm.size()) == cols);

    for (auto& entry : entries_)
    {
        std::get<0>(entry) = row_perm[std::get<0>(entry)];
        std::get<1>(entry) = col_perm[std::get<1>(entry)];
    }
}

template <typename T>
template <typename U>
SparseMatrix<U> CooMatrix<T>::ToSparse() const
{
    int rows;
    int cols;
    std::tie(rows, cols) = FindSize();

    if (entries_.empty())
    {
        return SparseMatrix<U>(rows, cols);
    }

    std::vector<int> indptr(rows + 1, 0);

    std::vector<std::unordered_map<int, int>> index_map(rows);

    for (const auto& entry : entries_)
    {
        int row = std::get<0>(entry);
        int col = std::get<1>(entry);

        auto search = index_map[row].find(col);

        if (search == index_map[row].end())
        {
            indptr[row + 1]++;
            int size = index_map[row].size();
            index_map[row][col] = size;
        }
    }

    std::partial_sum(std::begin(indptr), std::end(indptr), std::begin(indptr));

    const size_t nnz = indptr.back();

    std::vector<int> indices(nnz);
    std::vector<U> data(nnz, (T)0);

    for (const auto& tup : entries_)
    {
        const int i = std::get<0>(tup);
        const int j = std::get<1>(tup);
        const U val = std::get<2>(tup);

        int index = indptr[i] + index_map[i].at(j);
        indices[index] = j;
        data[index] += val;
    }

    return SparseMatrix<U>(std::move(indptr), std::move(indices), std::move(data), rows, cols);
}

template <typename T>
void CooMatrix<T>::Mult(const VectorView<double>& input, VectorView<double> output) const
{
    assert(Rows() == output.size());
    assert(Cols() == input.size());

    output = 0;

    for (const auto& entry : entries_)
    {
        const int i = std::get<0>(entry);
        const int j = std::get<1>(entry);
        const double val = std::get<2>(entry);

        output[i] += val * input[j];
    }
}

template <typename T>
void CooMatrix<T>::MultAT(const VectorView<double>& input, VectorView<double> output) const
{
    assert(Rows() == output.size());
    assert(Cols() == input.size());

    output = 0;

    for (const auto& entry : entries_)
    {
        const int i = std::get<0>(entry);
        const int j = std::get<1>(entry);
        const double val = std::get<2>(entry);

        output[j] += val * input[i];
    }
}

template <typename T>
void CooMatrix<T>::Print(const std::string& label, std::ostream& out) const
{
    out << label << "\n";

    for (const auto& entry : entries_)
    {
        const int i = std::get<0>(entry);
        const int j = std::get<1>(entry);
        const T val = std::get<2>(entry);

        out << "(" << i << ", " << j << ") " << val << "\n";
    }

    out << "\n";
}

template <typename T>
void CooMatrix<T>::EliminateZeros(double tolerance)
{
    auto compare = [&] (const std::tuple<int, int, T>& entry)
    {
        const double val = std::get<2>(entry);
        return std::abs(val) < tolerance;
    };

    entries_.erase(std::remove_if(std::begin(entries_), std::end(entries_),
                   compare), std::end(entries_));
}

template <typename T>
std::tuple<int, int> CooMatrix<T>::FindSize() const
{
    if (size_set_)
    {
        return std::tuple<int, int> {rows_, cols_};
    }

    int rows = 0;
    int cols = 0;

    for (const auto& entry : entries_)
    {
        const int i = std::get<0>(entry);
        const int j = std::get<1>(entry);

        rows = std::max(rows, i);
        cols = std::max(cols, j);
    }

    return std::tuple<int, int> {rows + 1, cols + 1};
}

} // namespace linalgcpp

#endif // COOMATRIX_HPP__
