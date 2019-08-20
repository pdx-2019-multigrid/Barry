/*! @file */

#ifndef SPARSEMATRIX_HPP__
#define SPARSEMATRIX_HPP__

#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <assert.h>

#include "operator.hpp"
#include "densematrix.hpp"

namespace linalgcpp
{

/*! @brief Sparse matrix in CSR format

    3 arrays keep track of the nonzero entries:
        - indptr is the row pointer such that indptr[i] points
          to the start of row i in indices and data
        - indices is the column index of the entry
        - data is the value of the entry
*/
template <typename T = double>
class SparseMatrix : public Operator
{
    public:
        /*! @brief Default Constructor of zero size */
        SparseMatrix();

        /*! @brief Empty Constructor with set size
            @param size rows and columns in matrix
        */
        explicit SparseMatrix(int size);

        /*! @brief Empty Constructor with set size
            @param rows the number of rows
            @param cols the number of columns
        */
        SparseMatrix(int rows, int cols);

        /*! @brief Constructor setting the individual arrays and size
            @param indptr row pointer array
            @param indices column indices array
            @param data entry value array
            @param rows number of rows in the matrix
            @param cols number of cols in the matrix
        */
        SparseMatrix(std::vector<int> indptr,
                     std::vector<int> indices,
                     std::vector<T> data,
                     int rows, int cols);

        /*! @brief Diagonal Constructor
            @param diag values for the diagonal
        */
        explicit SparseMatrix(std::vector<T> diag);

        /*! @brief Copy Constructor */
        SparseMatrix(const SparseMatrix<T>& other) noexcept;

        /*! @brief Move constructor */
        SparseMatrix(SparseMatrix<T>&& other) noexcept;

        /*! @brief Destructor */
        ~SparseMatrix() noexcept = default;

        /*! @brief Sets this matrix equal to another
            @param other the matrix to copy
        */
        SparseMatrix<T>& operator=(SparseMatrix<T> other) noexcept;

        /*! @brief Swap two matrices
            @param lhs left hand side matrix
            @param rhs right hand side matrix
        */
        template <typename U>
        friend void swap(SparseMatrix<U>& lhs, SparseMatrix<U>& rhs) noexcept;

        /*! @brief The number of nonzero entries in this matrix
            @retval the nonzero entries of columns

            @note this includes explicit zeros
        */
        int nnz() const;

        /*! @brief Get the const row pointer array
            @retval the row pointer array
        */
        const std::vector<int>& GetIndptr() const;

        /*! @brief Get the const column indices
            @retval the column indices array
        */
        const std::vector<int>& GetIndices() const;

        /*! @brief Get the const entry values
            @retval the data array
        */
        const std::vector<T>& GetData() const;

        /*! @brief Get the row pointer array
            @retval the row pointer array
        */
        std::vector<int>& GetIndptr();

        /*! @brief Get the column indices
            @retval the column indices array
        */
        std::vector<int>& GetIndices();

        /*! @brief Get the entry values
            @retval the data array
        */
        std::vector<T>& GetData();

        /*! @brief Get the indices from one row
            @param row the row to get
            @retval indices the indices from one row
        */
        std::vector<int> GetIndices(int row) const;

        /*! @brief Get view of the indices from one row
            @param row the row to get
            @retval indices the indices from one row
        */
        VectorView<int> GetIndicesView(int row);

        /*! @brief Get view of the indices from one row
            @param row the row to get
            @retval indices the indices from one row
        */
        const VectorView<int> GetIndicesView(int row) const;

        /*! @brief Get the entries from one row
            @param row the row to get
            @retval the data from one row
        */
        std::vector<T> GetData(int row) const;

        /*! @brief Get the number of entries in a row
            @param row the row to get
            @retval int the number of entries in the row
        */
        int RowSize(int row) const;

        /*! @brief Print the nonzero entries as a list
            @param label the label to print before the list of entries
            @param out stream to print to
        */
        void Print(const std::string& label = "", std::ostream& out = std::cout) const;

        /*! @brief Print the entries of this matrix in dense format
            @param label the label to print before the list of entries
            @param out stream to print to
        */
        void PrintDense(const std::string& label = "", std::ostream& out = std::cout) const;

        /*! @brief Generate a dense version of this matrix
            @retval the dense version of this matrix
        */
        DenseMatrix ToDense() const;

        /*! @brief Generate a dense version of this matrix
            @param dense holds the dense version of this matrix
        */
        void ToDense(DenseMatrix& dense) const;

        /*! @brief Sort the column indices in each row */
        void SortIndices();

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @retval output the output vector y
        */
        template <typename U = T>
        Vector<T> Mult(const VectorView<U>& input) const;

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @retval output the output vector y
        */
        template <typename U = T>
        Vector<T> MultAT(const VectorView<U>& input) const;

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @param output the output vector y
        */
        template <typename U = T, typename V = T>
        void Mult(const VectorView<U>& input, VectorView<V> output) const;

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @param output the output vector y
        */
        template <typename U = T, typename V = T>
        void MultAT(const VectorView<U>& input, VectorView<V> output) const;

        /*! @brief Multiplies a dense matrix: AB = C
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        DenseMatrix Mult(const DenseMatrix& input) const;

        /*! @brief Multiplies a dense matrix by the transpose
            of this matrix: A^T B = C
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        DenseMatrix MultAT(const DenseMatrix& input) const;

        /*! @brief Multiplies a dense matrix and stores the result transposed: A B = C^T
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        DenseMatrix MultCT(const DenseMatrix& input) const;

        /*! @brief Multiplies a dense matrix: AB = C
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        void Mult(const DenseMatrix& input, DenseMatrix& output) const;

        /*! @brief Multiplies a dense matrix by the transpose
            of this matrix: A^T B = C
            @param input the input dense matrix B
            @param output the output dense matrix C
        */
        void MultAT(const DenseMatrix& input, DenseMatrix& output) const;

        /*! @brief Multiplies a dense matrix and stores the result transposed: A B = C^T
            @param input the input dense matrix B
            @param output the output dense matrix C
        */
        void MultCT(const DenseMatrix& input, DenseMatrix& output) const;

        /*! @brief Multiplies a sparse matrix: AB = C
            @param rhs the input sparse matrix B
            @retval the output sparse matrix C
        */
        template <typename U = T, typename V = typename std::common_type<T, U>::type>
        SparseMatrix<V> Mult(const SparseMatrix<U>& rhs) const;

        /*! @brief Multiplies a sparse matrix: AB = C
            @param rhs the input sparse matrix B
            @param marker reusable temp work space
            @retval the output sparse matrix C
        */
        template <typename U = T, typename V = typename std::common_type<T, U>::type>
        SparseMatrix<V> Mult(const SparseMatrix<U>& rhs, std::vector<int>& marker) const;

        /*! @brief Genereates the transpose of this matrix*/
        template <typename U = T>
        SparseMatrix<U> Transpose() const;

        /*! @brief Genereates the transpose of this matrix into
                   dense matrix
            @returns output Dense transpose
        */
        DenseMatrix TransposeDense() const;

        /*! @brief Generates the transpose of this matrix into
                   dense matrix
            @param output Dense transpose
        */
        void TransposeDense(DenseMatrix& output) const;









        Vector<T> Jacobi(Vector<T>) const;
        Vector<T> L1(Vector<T>) const;

        template <typename U = double>
        Vector<U> GaussSeidel(Vector<U>) const;

        template <typename U = double> 
        Vector<U> ForwardGauss(Vector<U> r) const;

        template <typename U = double> 
        Vector<U> BackwardGauss(Vector<U> r) const;









        /*! @brief Get the diagonal entries
            @retval the diagonal entries
        */
        std::vector<double> GetDiag() const;

        /*! @brief Add to the diagonal
            @param diag the diagonal entries
        */
        void AddDiag(const std::vector<T>& diag);

        /*! @brief Add to the diagonal
            @param diag the diagonal entries
        */
        void AddDiag(T val);

        /*! @brief Extract a submatrix out of this matrix
            @param rows the rows to extract
            @param cols the columns to extract
            @retval the submatrix

            @note a workspace array the size of the number of columns
                 in this matrix is used for this routine. If multiple
                 extractions are need from large matrices, it is
                 recommended to reuse a single marker array to avoid
                 large memory allocations.
        */
        SparseMatrix<T> GetSubMatrix(const std::vector<int>& rows,
                                     const std::vector<int>& cols) const;

        /*! @brief Extract a submatrix out of this matrix
            @param rows the rows to extract
            @param cols the columns to extract
            @param marker workspace used to keep track of column indices,
                  must be at least the size of the number of columns
            @retval the submatrix
        */
        SparseMatrix<T> GetSubMatrix(const std::vector<int>& rows,
                                     const std::vector<int>& cols,
                                     std::vector<int>& marker) const;

        /*! @brief Multiply by a scalar */
        template <typename U = T>
        SparseMatrix<T>& operator*=(U val);

        /*! @brief Divide by a scalar */
        template <typename U = T>
        SparseMatrix<T>& operator/=(U val);

        /*! @brief Set all nonzeros to a scalar */
        template <typename U = T>
        SparseMatrix<T>& operator=(U val);

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @retval output the output vector y
        */
        template <typename U = T>
        Vector<U> operator*(const VectorView<U>& input) const;

        /*! @brief Sum of all data
            @retval sum Sum of all data
        */
        T Sum() const;

        /*! @brief Sum of absolute value of all data
            @retval sum Sum of absolute value of all data
        */
        T MaxNorm() const;

        /*! @brief Scale rows by diagonal matrix
            @param values scale per row
        */
        void ScaleRows(const SparseMatrix<T>& values);

        /*! @brief Scale cols by diagonal matrix
            @param values scale per cols
        */
        void ScaleCols(const SparseMatrix<T>& values);

        /*! @brief Scale rows by inverse of diagonal matrix
            @param values scale per row
        */
        void InverseScaleRows(const SparseMatrix<T>& values);

        /*! @brief Scale cols by inverse of diagonal matrix
            @param values scale per cols
        */
        void InverseScaleCols(const SparseMatrix<T>& values);

        /*! @brief Scale rows by given values
            @param values scale per row
        */
        void ScaleRows(const std::vector<T>& values);

        /*! @brief Scale cols by given values
            @param values scale per cols
        */
        void ScaleCols(const std::vector<T>& values);

        /*! @brief Scale rows by inverse of given values
            @param values scale per row
        */
        void InverseScaleRows(const std::vector<T>& values);

        /*! @brief Scale cols by inverse of given values
            @param values scale per cols
        */
        void InverseScaleCols(const std::vector<T>& values);

        /*! @brief Permute the columns
            @param perm permutation to apply
        */
        void PermuteCols(const std::vector<int>& perm);

        /*! @brief Eliminate a row and column and replace diagonal
                   value by 1
            @param index row and column to eliminate
        */
        void EliminateRowCol(int index);

        /*! @brief Eliminate all marked rows and columns from square matrix.
            @param marker indexable type where marker[i] is elimnated if true

            @note If diagonal entry exists, it is set to 1.0
        */
        template <typename U>
        void EliminateRowCol(const U& marker);

        /*! @brief Eliminate a row by setting all row entries to zero
            @param index row to eliminate
        */
        void EliminateRow(int index);

        /*! @brief Eliminate a column by setting all column entries to zero
            @param index column to eliminate
        */
        void EliminateCol(int index);

        /*! @brief Eliminate all marked columns
            @param marker indexable type where marker[i] is elimnated if true
        */
        template <typename U>
        void EliminateCol(const U& marker);

        /*! @brief Eliminate all marked columns, setting b[i] = -A_ij * x[j]
            @param marker indexable type where marker[i] is elimnated if true
        */
        template <typename U>
        void EliminateCol(const U& marker, const VectorView<double>& x, VectorView<double> b);

        /*! @brief Remove entries less than a tolerance
            @param tol tolerance to remove
        */
        void EliminateZeros(T tolerance = 0);

        /// Operator Requirement, calls the templated Mult
        void Mult(const VectorView<double>& input, VectorView<double> output) const override;
        /// Operator Requirement, calls the templated MultAT
        void MultAT(const VectorView<double>& input, VectorView<double> output) const override;

        using Operator::Mult;
        using Operator::MultAT;

    private:
        std::vector<int> indptr_;
        std::vector<int> indices_;
        std::vector<T> data_;
};

template <typename T>
SparseMatrix<T>::SparseMatrix()
    : indptr_{0}, indices_(0), data_(0)
{

}

template <typename T>
SparseMatrix<T>::SparseMatrix(int size)
    : SparseMatrix<T>(size, size)
{

}

template <typename T>
SparseMatrix<T>::SparseMatrix(int rows, int cols)
    : Operator(rows, cols),
      indptr_(std::vector<int>(rows + 1, 0)), indices_(0), data_(0)
{

}

template <typename T>
SparseMatrix<T>::SparseMatrix(std::vector<int> indptr,
                              std::vector<int> indices,
                              std::vector<T> data,
                              int rows, int cols)
    : Operator(rows, cols),
      indptr_(std::move(indptr)), indices_(std::move(indices)), data_(std::move(data))
{
    assert(indptr_.size() == rows_ + 1u);
    assert(indices_.size() == data_.size());
    assert(indptr_[0] == 0);

    if (indices_.size() > 0)
    {
        assert(*std::max_element(std::begin(indices_), std::end(indices_)) < static_cast<int>(cols_));
        assert(*std::min_element(std::begin(indices_), std::end(indices_)) >= 0);
    }
}

template <typename T>
SparseMatrix<T>::SparseMatrix(std::vector<T> diag)
    : Operator(diag.size()),
      indptr_(diag.size() + 1), indices_(diag.size()), data_(std::move(diag))
{
    std::iota(begin(indptr_), end(indptr_), 0);
    std::iota(begin(indices_), end(indices_), 0);
}

template <typename T>
SparseMatrix<T>::SparseMatrix(const SparseMatrix<T>& other) noexcept
    : Operator(other),
      indptr_(other.indptr_), indices_(other.indices_), data_(other.data_)
{
}

template <typename T>
SparseMatrix<T>::SparseMatrix(SparseMatrix<T>&& other) noexcept
{
    swap(*this, other);
}

template <typename T>
SparseMatrix<T>& SparseMatrix<T>::operator=(SparseMatrix<T> other) noexcept
{
    swap(*this, other);

    return *this;
}

template <typename T>
void swap(SparseMatrix<T>& lhs, SparseMatrix<T>& rhs) noexcept
{
    swap(static_cast<Operator&>(lhs), static_cast<Operator&>(rhs));

    std::swap(lhs.indptr_, rhs.indptr_);
    std::swap(lhs.indices_, rhs.indices_);
    std::swap(lhs.data_, rhs.data_);
}

template <typename T>
void SparseMatrix<T>::Print(const std::string& label, std::ostream& out) const
{
    constexpr int width = 6;

    out << label << "\n";

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            out << "(" << i << ", " << indices_[j] << ") "
                << std::setw(width) << data_[j] << "\n";
        }
    }

    out << "\n";
}

template <typename T>
void SparseMatrix<T>::PrintDense(const std::string& label, std::ostream& out) const
{
    const DenseMatrix dense = ToDense();

    dense.Print(label, out);
}

template <typename T>
DenseMatrix SparseMatrix<T>::ToDense() const
{
    DenseMatrix dense(rows_, cols_);

    ToDense(dense);

    return dense;
}

template <typename T>
void SparseMatrix<T>::ToDense(DenseMatrix& dense) const
{
    dense.SetSize(rows_, cols_);
    dense = 0.0;

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            dense(i, indices_[j]) = data_[j];
        }
    }
}

template <typename T>
void SparseMatrix<T>::SortIndices()
{
    const auto compare_cols = [&](int i, int j)
    {
        return indices_[i] < indices_[j];
    };

    std::vector<int> permutation(indices_.size());
    std::iota(begin(permutation), end(permutation), 0);

    for (int i = 0; i < rows_; ++i)
    {
        const int start = indptr_[i];
        const int end = indptr_[i + 1];

        std::sort(begin(permutation) + start,
                  begin(permutation) + end,
                  compare_cols);
    }

    std::vector<int> sorted_indices(indices_.size());
    std::vector<T> sorted_data(data_.size());

    std::transform(begin(permutation), end(permutation), begin(sorted_indices),
                   [&] (int i)
    {
        return indices_[i];
    });
    std::transform(begin(permutation), end(permutation), begin(sorted_data),
                   [&] (int i)
    {
        return data_[i];
    });

    std::swap(indices_, sorted_indices);
    std::swap(data_, sorted_data);
}

template <typename T>
DenseMatrix SparseMatrix<T>::Mult(const DenseMatrix& input) const
{
    DenseMatrix output(rows_, input.Cols());
    Mult(input, output);

    return output;
}

template <typename T>
DenseMatrix SparseMatrix<T>::MultAT(const DenseMatrix& input) const
{
    DenseMatrix output(cols_, input.Cols());
    MultAT(input, output);

    return output;
}

template <typename T>
DenseMatrix SparseMatrix<T>::MultCT(const DenseMatrix& input) const
{
    DenseMatrix output(input.Cols(), rows_);
    MultCT(input, output);

    return output;
}

template <typename T>
void SparseMatrix<T>::Mult(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(input.Rows() == cols_);

    output.SetSize(rows_, input.Cols());
    output = 0.0;

    for (int k = 0; k < input.Cols(); ++k)
    {
        for (int i = 0; i < rows_; ++i)
        {
            double val = 0.0;

            for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
            {
                val += data_[j] * input(indices_[j], k);
            }

            output(i, k) = val;
        }
    }
}

template <typename T>
void SparseMatrix<T>::MultAT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(input.Rows() == rows_);

    output.SetSize(cols_, input.Cols());
    output = 0.0;

    for (int k = 0; k < input.Cols(); ++k)
    {
        for (int i = 0; i < rows_; ++i)
        {
            for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
            {
                output(indices_[j], k) += data_[j] * input(i, k);
            }
        }
    }
}

template <typename T>
void SparseMatrix<T>::MultCT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(input.Rows() == cols_);

    output.SetSize(input.Cols(), rows_);
    output = 0.0;

    for (int k = 0; k < input.Cols(); ++k)
    {
        for (int i = 0; i < rows_; ++i)
        {
            double val = 0.0;

            for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
            {
                val += data_[j] * input(indices_[j], k);
            }

            output(k, i) = val;
        }
    }
}

template <typename T>
template <typename U>
SparseMatrix<U> SparseMatrix<T>::Transpose() const
{
    std::vector<int> out_indptr(cols_ + 1, 0);
    std::vector<int> out_indices(nnz());
    std::vector<U> out_data(nnz());

    for (const int& col : indices_)
    {
        out_indptr[col + 1]++;
    }

    for (int i = 0; i < cols_; ++i)
    {
        out_indptr[i + 1] += out_indptr[i];
    }

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            const int row = indices_[j];
            const T val = data_[j];

            out_indices[out_indptr[row]] = i;
            out_data[out_indptr[row]] = val;
            out_indptr[row]++;
        }
    }

    for (int i = cols_; i > 0; --i)
    {
        out_indptr[i] = out_indptr[i - 1];
    }

    out_indptr[0] = 0;

    return SparseMatrix<U>(std::move(out_indptr), std::move(out_indices), std::move(out_data),
                           cols_, rows_);
}

template <typename T>
DenseMatrix SparseMatrix<T>::TransposeDense() const
{
    DenseMatrix output(cols_, rows_);

    TransposeDense(output);

    return output;
}

template <typename T>
void SparseMatrix<T>::TransposeDense(DenseMatrix& output) const
{
    output.SetSize(cols_, rows_);
    output = 0.0;

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            output(indices_[j], i) = data_[j];
        }
    }
}










template <typename T>
Vector<T> SparseMatrix<T>::Jacobi(Vector<T> r) const
{
  assert(rows_ == cols_);
  assert(rows_ == r.size());

  Vector<double> diag(this->GetDiag());
  for (int i = 0; i < rows_; ++i)
    r[i] = r[i] / diag[i];
  return r;
}

template <typename T>
Vector<T> SparseMatrix<T>::L1(Vector<T> r) const
{
  assert(rows_ == cols_);

  Vector<double> diag(this->GetDiag());
  for (int i = 0; i < rows_; ++i)
  {
    double sum = 0.0;
    for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
      sum += abs(data_[j]) * sqrt(diag[i] / diag[indices_[j]]);
    r[i] = r[i] / sum;
  }
  return r;
}

template <typename T>
template <typename U>
Vector<U> SparseMatrix<T>::GaussSeidel(Vector<U> r) const
{
  assert(rows_ == cols_);
  assert(rows_ == r.size());
  
  const int n = cols_;
  double sum;
  Vector<double> diag(n);
  Vector<double> y(n);
    
  // Forward substitution to solve: (D + L)y = r.
  for (int i = 0; i < n; ++i)
  {
    sum = 0.0;
    for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
    {
      if (i == indices_[j])
        diag[i] = data_[j];
        
      if (indices_[j] < i)
        sum += y[indices_[j]] * data_[j];
    }
    y[i] = (r[i] - sum) / diag[i];
  }

  y *= diag; // Let y be diag * y.

  // Backward substitution. Vectors r and y switch roles.
  // Solving the equation: (D + U)r = y.

  for (int i = n - 1; i > -1; --i)
  {
    sum = 0.0;
    for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
    {
      if (indices_[j] > i)
        sum += r[indices_[j]] * data_[j];
    }
    r[i] = (y[i] - sum) / diag[i];
  } 
  return r;
}

template <typename T>
template <typename U>
Vector<U> SparseMatrix<T>::ForwardGauss(Vector<U> r) const
{
  const int n = cols_;
  double sum;
  double diag;
  Vector<double> y(n);

  for (int i = 0; i < n; ++i)
  {
    sum = 0.0;
    for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
    {
      if (i == indices_[j])
        diag = data_[j];
        
      if (indices_[j] < i)
        sum += y[indices_[j]] * data_[j];
    }
    y[i] = (r[i] - sum) / diag;
  }
  return y;
}

template <typename T>
template <typename U>
Vector<U> SparseMatrix<T>::BackwardGauss(Vector<U> r) const
{
  const int n = cols_;
  double sum;
  
  double diag;
  Vector<double> y(n);

  for (int i = n - 1; i > -1; --i)
  {
    sum = 0.0;
    for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
    {
      if (i == indices_[j])
        diag = data_[j];
        
      if (indices_[j] > i)
        sum += y[indices_[j]] * data_[j];
    }
    y[i] = (r[i] - sum) / diag;
  }
  return y;
}







template <typename T>
std::vector<double> SparseMatrix<T>::GetDiag() const
{
    assert(rows_ == cols_);

    std::vector<double> diag(rows_);

    for (int i = 0; i < rows_; ++i)
    {
        double val = 0.0;

        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            if (indices_[j] == i)
            {
                val = data_[j];
            }
        }

        diag[i] = val;
    }

    return diag;
}

template <typename T>
void SparseMatrix<T>::AddDiag(const std::vector<T>& diag)
{
    assert(rows_ == cols_);

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            if (indices_[j] == i)
            {
                data_[j] += diag[i];
            }
        }
    }
}

template <typename T>
void SparseMatrix<T>::AddDiag(T val)
{
    assert(rows_ == cols_);

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            if (indices_[j] == i)
            {
                data_[j] += val;
            }
        }
    }
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::GetSubMatrix(const std::vector<int>& rows,
                                              const std::vector<int>& cols) const
{
    std::vector<int> marker(cols_, -1);

    return GetSubMatrix(rows, cols, marker);
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::GetSubMatrix(const std::vector<int>& rows,
                                              const std::vector<int>& cols,
                                              std::vector<int>& marker) const
{
    assert(static_cast<int>(marker.size()) >= cols_);

    std::vector<int> out_indptr(rows.size() + 1);
    out_indptr[0] = 0;

    int out_nnz = 0;

    const int out_rows = rows.size();
    const int out_cols = cols.size();

    for (int i = 0; i < out_cols; ++i)
    {
        marker[cols[i]] = i;
    }

    for (int i = 0; i < out_rows; ++i)
    {
        const int row = rows[i];

        assert(row < rows_);

        for (int j = indptr_[row]; j < indptr_[row + 1]; ++j)
        {
            if (marker[indices_[j]] != -1)
            {
                ++out_nnz;
            }
        }

        out_indptr[i + 1] = out_nnz;
    }

    std::vector<int> out_indices(out_nnz);
    std::vector<T> out_data(out_nnz);

    int total = 0;

    for (auto row : rows)
    {
        for (int j = indptr_[row]; j < indptr_[row + 1]; ++j)
        {
            if (marker[indices_[j]] != -1)
            {
                out_indices[total] = marker[indices_[j]];
                out_data[total] = data_[j];

                total++;
            }
        }
    }

    for (auto i : cols)
    {
        marker[i] = -1;
    }

    return SparseMatrix<T>(std::move(out_indptr), std::move(out_indices), std::move(out_data),
                           out_rows, out_cols);
}

template <typename T>
inline
int SparseMatrix<T>::nnz() const
{
    return data_.size();
}

template <typename T>
inline
const std::vector<int>& SparseMatrix<T>::GetIndptr() const
{
    return indptr_;
}

template <typename T>
inline
const std::vector<int>& SparseMatrix<T>::GetIndices() const
{
    return indices_;
}

template <typename T>
inline
const std::vector<T>& SparseMatrix<T>::GetData() const
{
    return data_;
}

template <typename T>
inline
std::vector<int>& SparseMatrix<T>::GetIndptr()
{
    return indptr_;
}

template <typename T>
inline
std::vector<int>& SparseMatrix<T>::GetIndices()
{
    return indices_;
}

template <typename T>
inline
std::vector<T>& SparseMatrix<T>::GetData()
{
    return data_;
}

template <typename T>
inline
std::vector<int> SparseMatrix<T>::GetIndices(int row) const
{
    assert(row >= 0 && row < rows_);

    const int start = indptr_[row];
    const int end = indptr_[row + 1];

    return std::vector<int>(begin(indices_) + start, begin(indices_) + end);
}

template <typename T>
inline
VectorView<int> SparseMatrix<T>::GetIndicesView(int row)
{
    assert(row >= 0 && row < rows_);

    const int start = indptr_[row];
    const int end = indptr_[row + 1];
    const int size = end - start;

    int* data = indices_.data() + start;

    return VectorView<int>{data, size};
}

template <typename T>
inline
const VectorView<int> SparseMatrix<T>::GetIndicesView(int row) const
{
    assert(row >= 0 && row < rows_);

    const int start = indptr_[row];
    const int end = indptr_[row + 1];
    const int size = end - start;

    int* data = const_cast<int*>(indices_.data()) + start;

    return VectorView<int>{data, size};
}

template <typename T>
inline
std::vector<T> SparseMatrix<T>::GetData(int row) const
{
    assert(row >= 0 && row < rows_);

    const int start = indptr_[row];
    const int end = indptr_[row + 1];

    return std::vector<T>(begin(data_) + start, begin(data_) + end);
}

template <typename T>
template <typename U>
Vector<T> SparseMatrix<T>::Mult(const VectorView<U>& input) const
{
    Vector<T> output(rows_);
    Mult(input, output);

    return output;
}

template <typename T>
template <typename U>
Vector<T> SparseMatrix<T>::MultAT(const VectorView<U>& input) const
{
    Vector<T> output(cols_);
    MultAT(input, output);

    return output;
}

template <typename T>
template <typename U, typename V>
void SparseMatrix<T>::Mult(const VectorView<U>& input, VectorView<V> output) const
{
    assert(input.size() == cols_);
    assert(output.size() == rows_);

    for (int i = 0; i < rows_; ++i)
    {
        V val = 0;

        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            val += data_[j] * input[indices_[j]];
        }

        output[i] = val;
    }
}

template <typename T>
template <typename U, typename V>
void SparseMatrix<T>::MultAT(const VectorView<U>& input, VectorView<V> output) const
{
    assert(input.size() == rows_);
    assert(output.size() == cols_);

    std::fill(std::begin(output), std::end(output), 0.0);

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            output[indices_[j]] += data_[j] * input[i];
        }
    }
}

template <typename T>
void SparseMatrix<T>::Mult(const VectorView<double>& input, VectorView<double> output) const
{
    Mult<double, double>(input, output);
}

template <typename T>
void SparseMatrix<T>::MultAT(const VectorView<double>& input, VectorView<double> output) const
{
    MultAT<double, double>(input, output);
}

template <typename T>
template <typename U>
Vector<U> SparseMatrix<T>::operator*(const VectorView<U>& input) const
{
    return Mult<U>(input);
}

template <typename T>
template <typename U, typename V>
SparseMatrix<V> SparseMatrix<T>::Mult(const SparseMatrix<U>& rhs) const
{
    std::vector<int> marker(rhs.Cols());

    return Mult<U, V>(rhs, marker);
}

template <typename T>
template <typename U, typename V>
SparseMatrix<V> SparseMatrix<T>::Mult(const SparseMatrix<U>& rhs, std::vector<int>& marker) const
{
    assert(rhs.Rows() == cols_);

    marker.resize(rhs.Cols());
    std::fill(begin(marker), end(marker), -1);

    std::vector<int> out_indptr(rows_ + 1);
    out_indptr[0] = 0;

    int out_nnz = 0;

    const std::vector<int>& rhs_indptr = rhs.GetIndptr();
    const std::vector<int>& rhs_indices = rhs.GetIndices();
    const std::vector<U>& rhs_data = rhs.GetData();

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            for (int k = rhs_indptr[indices_[j]]; k < rhs_indptr[indices_[j] + 1]; ++k)
            {
                if (marker[rhs_indices[k]] != static_cast<int>(i))
                {
                    marker[rhs_indices[k]] = i;
                    ++out_nnz;
                }
            }
        }

        out_indptr[i + 1] = out_nnz;
    }

    std::fill(begin(marker), end(marker), -1);

    std::vector<int> out_indices(out_nnz);
    std::vector<V> out_data(out_nnz);

    int total = 0;

    for (int i = 0; i < rows_; ++i)
    {
        int row_nnz = total;

        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            for (int k = rhs_indptr[indices_[j]]; k < rhs_indptr[indices_[j] + 1]; ++k)
            {
                if (marker[rhs_indices[k]] < row_nnz)
                {
                    marker[rhs_indices[k]] = total;
                    out_indices[total] = rhs_indices[k];
                    out_data[total] = data_[j] * rhs_data[k];

                    total++;
                }
                else
                {
                    out_data[marker[rhs_indices[k]]] += data_[j] * rhs_data[k];
                }
            }
        }
    }

    return SparseMatrix<V>(std::move(out_indptr), std::move(out_indices), std::move(out_data),
                            rows_, rhs.Cols());
}

template <typename T>
template <typename U>
SparseMatrix<T>& SparseMatrix<T>::operator*=(U val)
{
    for (auto& i : data_)
    {
        i *= val;
    }

    return *this;
}

template <typename T>
template <typename U>
SparseMatrix<T>& SparseMatrix<T>::operator/=(U val)
{
    assert(val != 0);

    for (auto& i : data_)
    {
        i /= val;
    }

    return *this;
}

template <typename T>
template <typename U>
SparseMatrix<T>& SparseMatrix<T>::operator=(U val)
{
    assert(val != 0);

    std::fill(begin(data_), end(data_), val);

    return *this;
}

template <typename T>
int SparseMatrix<T>::RowSize(int row) const
{
    assert(row >= 0 && row < rows_);

    return indptr_[row + 1] - indptr_[row];
}

/*! @brief Multiply a sparse matrix and
           a scalar into a new sparse matrix: Aa = B
    @param lhs the left hand side matrix A
    @param val the right hand side scalar a
    @retval the multiplied matrix B
*/
template <typename U, typename V>
SparseMatrix<U> operator*(SparseMatrix<U> lhs, V val)
{
    return lhs *= val;
}

/*! @brief Multiply a sparse matrix and
           a scalar into a new sparse matrix: aA = B
    @param val the left hand side scalar a
    @param rhs the right hand side matrix A
    @retval the multiplied matrix B
*/
template <typename U, typename V>
SparseMatrix<U> operator*(V val, SparseMatrix<U> rhs)
{
    return rhs *= val;
}

template <typename T>
T SparseMatrix<T>::Sum() const
{
    T sum = std::accumulate(std::begin(data_), std::end(data_), (T)0);
    return sum;
}

template <typename T>
T SparseMatrix<T>::MaxNorm() const
{
    T sum = 0;

    for (const auto& val : data_)
    {
        sum += std::abs(val);
    }

    return sum;
}

template <typename T>
void SparseMatrix<T>::ScaleRows(const SparseMatrix<T>& values)
{
    ScaleRows(values.GetData());
}

template <typename T>
void SparseMatrix<T>::ScaleCols(const SparseMatrix<T>& values)
{
    ScaleCols(values.GetData());
}

template <typename T>
void SparseMatrix<T>::InverseScaleRows(const SparseMatrix<T>& values)
{
    InverseScaleRows(values.GetData());
}

template <typename T>
void SparseMatrix<T>::InverseScaleCols(const SparseMatrix<T>& values)
{
    InverseScaleCols(values.GetData());
}

template <typename T>
void SparseMatrix<T>::ScaleRows(const std::vector<T>& values)
{
    assert(static_cast<int>(values.size()) == rows_);

    for (int i = 0; i < rows_; ++i)
    {
        const T scale = values[i];

        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            data_[j] *= scale;
        }
    }
}

template <typename T>
void SparseMatrix<T>::ScaleCols(const std::vector<T>& values)
{
    assert(static_cast<int>(values.size()) == cols_);

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            data_[j] *= values[indices_[j]];
        }
    }
}

template <typename T>
void SparseMatrix<T>::InverseScaleRows(const std::vector<T>& values)
{
    assert(static_cast<int>(values.size()) == rows_);

    for (int i = 0; i < rows_; ++i)
    {
        const T scale = values[i];
        assert(scale != (T)0);

        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            data_[j] /= scale;
        }
    }
}

template <typename T>
void SparseMatrix<T>::InverseScaleCols(const std::vector<T>& values)
{
    assert(static_cast<int>(values.size()) == cols_);

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            auto scale = values[indices_[j]];
            assert(scale != (T)0);

            data_[j] /= scale;
        }
    }
}

template <typename T>
void SparseMatrix<T>::PermuteCols(const std::vector<int>& perm)
{
    assert(static_cast<int>(perm.size()) == cols_);

    int nnz = indices_.size();

    for (int i = 0; i < nnz; ++i)
    {
        assert(perm[indices_[i]] >= 0);
        assert(perm[indices_[i]] < cols_);

        indices_[i] = perm[indices_[i]];
    }
}

template <typename T>
void SparseMatrix<T>::EliminateRowCol(int index)
{
    assert(index >= 0);
    assert(index < rows_);
    assert(index < cols_);

    for (int row = 0; row < rows_; ++row)
    {
        for (int j = indptr_[row]; j < indptr_[row + 1]; ++j)
        {
            int col = indices_[j];

            if (row == index || col == index)
            {
                data_[j] = (col == row) ? 1.0 : 0.0;
            }
        }
    }
}

template <typename T>
template <typename U>
void SparseMatrix<T>::EliminateRowCol(const U& marker)
{
    assert(rows_ == cols_);
    assert(marker.size() >= cols_);

    for (int row = 0; row < rows_; ++row)
    {
        for (int j = indptr_[row]; j < indptr_[row + 1]; ++j)
        {
            int col = indices_[j];

            if (marker[row] || marker[col])
            {
                data_[j] = (row == col) ? 1.0 : 0.0;
            }
        }
    }
}

template <typename T>
void SparseMatrix<T>::EliminateRow(int index)
{
    assert(index >= 0);
    assert(index < rows_);

    for (int j = indptr_[index]; j < indptr_[index + 1]; ++j)
    {
        data_[j] = 0.0;
    }
}

template <typename T>
void SparseMatrix<T>::EliminateCol(int index)
{
    assert(index >= 0);
    assert(index < cols_ || cols_ == 0);

    for (int row = 0; row < rows_; ++row)
    {
        for (int j = indptr_[row]; j < indptr_[row + 1]; ++j)
        {
            if (indices_[j] == index)
            {
                data_[j] = 0.0;
            }
        }
    }
}

template <typename T>
template <typename U>
void SparseMatrix<T>::EliminateCol(const U& marker)
{
    assert(marker.size() >= cols_);

    for (int row = 0; row < rows_; ++row)
    {
        for (int j = indptr_[row]; j < indptr_[row + 1]; ++j)
        {
            if (marker[indices_[j]])
            {
                data_[j] = 0.0;
            }
        }
    }
}

template <typename T>
template <typename U>
void SparseMatrix<T>::EliminateCol(const U& marker, const VectorView<double>& x, VectorView<double> b)
{
    assert(marker.size() >= cols_);

    for (int row = 0; row < rows_; ++row)
    {
        for (int j = indptr_[row]; j < indptr_[row + 1]; ++j)
        {
            if (marker[indices_[j]] && data_[j] != 0.0)
            {
                b[row] -= data_[j] * x[indices_[j]];
                data_[j] = 0.0;
            }
        }
    }
}

template <typename T>
void SparseMatrix<T>::EliminateZeros(T tolerance)
{
    std::vector<int> elim_indptr(1, 0);
    std::vector<int> elim_indices;
    std::vector<T> elim_data;

    elim_indptr.reserve(indptr_.size());
    elim_indices.reserve(indices_.size());
    elim_data.reserve(data_.size());

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = indptr_[i]; j < indptr_[i + 1]; ++j)
        {
            if (std::fabs(data_[j]) > tolerance)
            {
                elim_indices.push_back(indices_[j]);
                elim_data.push_back(data_[j]);
            }
        }

        elim_indptr.push_back(elim_indices.size());
    }

    std::swap(elim_indptr, indptr_);
    std::swap(elim_indices, indices_);
    std::swap(elim_data, data_);
}

} //namespace linalgcpp

#endif // SPARSEMATRIX_HPP__
