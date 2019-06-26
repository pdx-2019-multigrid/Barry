/*! @file */

#ifndef DENSEMATRIX_HPP__
#define DENSEMATRIX_HPP__

#include <memory>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include "operator.hpp"
#include "vector.hpp"

namespace linalgcpp
{

/*! @brief Dense matrix in column major format.
    This means data is arranged contiguously by column vectors.
*/
class DenseMatrix : public Operator
{
    public:
        /*! @brief Default Constructor of zero size */
        DenseMatrix();

        /*! @brief Square Constructor of setting the number of rows
            and columns
            @param size the number of rows and columns
        */
        explicit DenseMatrix(int size);

        /*! @brief Rectangle Constructor of setting the number of rows
            and columns
            @param rows the number of rows
            @param cols the number of columns

            @note values intitialed to zero
        */
        DenseMatrix(int rows, int cols);

        /*! @brief Rectangle Constructor of setting the number of rows,
            columns, and intital values
            @param rows the number of rows
            @param cols the number of columns
            @param data the initial data
        */
        DenseMatrix(int rows, int cols, std::vector<double> data);

        /*! @brief Copy Constructor */
        DenseMatrix(const DenseMatrix& other) noexcept;

        /*! @brief Move constructor */
        DenseMatrix(DenseMatrix&& other) noexcept;

        /*! @brief Set this matrix equal to other */
        DenseMatrix& operator=(DenseMatrix other) noexcept;

        /*! @brief Destructor */
        ~DenseMatrix() noexcept = default;

        /*! @brief Swap two matrices
            @param lhs left hand side matrix
            @param rhs right hand side matrix
        */
        friend void swap(DenseMatrix& lhs, DenseMatrix& rhs) noexcept;

        /*! @brief Get Data pointer */
        double* GetData();

        /*! @brief Get Data pointer */
        const double* GetData() const;

        /*! @brief Copies data to input array
            @param data copied data
        */
        void CopyData(std::vector<double>& data) const;

        /*! @brief Square Resizes Matrix
            @param size square size to set

            @warning new entries not initialized!
        */
        void SetSize(int size);

        /*! @brief Square Resizes Matrix and sets new values if 
                   new size is larger than previous
            @param size square size to set
            @param val values of new entries
        */
        void SetSize(int size, double val);

        /*! @brief Rectangle Resizes Matrix
            @param size square size to set

            @warning new entries not initialized!
        */
        void SetSize(int rows, int cols);

        /*! @brief Rectangle Resizes Matrix and sets new values if 
                   new size is larger than previous
            @param size square size to set
            @param val values of new entries
        */
        void SetSize(int rows, int cols, double val);


        /*! @brief Computes the sum of all entries
            @retval the sum of all entries
        */
        double Sum() const;

        /*! @brief Computes the maximum of all entries
            @retval the maximum of all entries
        */
        double Max() const;

        /*! @brief Computes the minimum of all entries
            @retval the minimum of all entries
        */
        double Min() const;

        /*! @brief Print the entries of this matrix in dense format
            @param label the label to print before the list of entries
            @param out stream to print to
            @param width total width of each entry including negative
            @param precision precision to print to
        */
        void Print(const std::string& label = "", std::ostream& out = std::cout, int width = 8, int precision = 4) const;

        /*! @brief Get the transpose of this matrix
            @retval transpose the tranpose
        */
        DenseMatrix Transpose() const;

        /*! @brief Store the transpose of this matrix
                   into the user provided matrix
            @param transpose transpose the tranpose

            @note Size must be set beforehand!
        */
        void Transpose(DenseMatrix& transpose) const;

        /*! @brief Index operator
            @param row row index
            @param col column index
            @retval a reference to the value at (i, j)
        */
        double& operator()(int row, int col);

        /*! @brief Const index operator
            @param row row index
            @param col column index
            @retval a const reference to the value at (i, j)
        */
        const double& operator()(int row, int col) const;

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @retval output the output vector y
        */
        template <typename T>
        Vector<double> Mult(const VectorView<T>& input) const;

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @retval output the output vector y
        */
        template <typename T>
        Vector<double> MultAT(const VectorView<T>& input) const;

        /*! @brief Multiplies a vector: Ax = y
            @param input the input vector x
            @param output the output vector y
        */
        template <typename T, typename U>
        void Mult(const VectorView<T>& input, VectorView<U> output) const;

        /*! @brief Multiplies a vector by the transpose
            of this matrix: A^T x = y
            @param input the input vector x
            @param output the output vector y
        */
        template <typename T, typename U>
        void MultAT(const VectorView<T>& input, VectorView<U> output) const;

        /*! @brief Multiplies a dense matrix: AB = C
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        DenseMatrix Mult(const DenseMatrix& input) const;

        /*! @brief Multiplies a dense matrix: A^T B = C
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        DenseMatrix MultAT(const DenseMatrix& input) const;

        /*! @brief Multiplies a dense matrix: AB^T = C
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        DenseMatrix MultBT(const DenseMatrix& input) const;

        /*! @brief Multiplies a dense matrix: A^T B^T = Y
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        DenseMatrix MultABT(const DenseMatrix& input) const;

        /*! @brief Multiplies a dense matrix: AB = C
            @param input the input dense matrix B
            @param output the output dense matrix C
        */
        void Mult(const DenseMatrix& input, DenseMatrix& output) const;

        /*! @brief Multiplies a dense matrix: A^T B = C
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        void MultAT(const DenseMatrix& input, DenseMatrix& output) const;

        /*! @brief Multiplies a dense matrix: AB^T = C
            @param input the input dense matrix B
            @retval output the output dense matrix C
        */
        void MultBT(const DenseMatrix& input, DenseMatrix& output) const;

        /*! @brief Multiplies a dense matrix: A^T B^T = C
            @param input the input dense matrix B
            @param output the output dense matrix C
        */
        void MultABT(const DenseMatrix& input, DenseMatrix& output) const;

        /*! @brief Adds the entries of another matrix to this one
            @param other the other dense matrix
        */
        DenseMatrix& operator+=(const DenseMatrix& other);

        /*! @brief Subtracts the entries of another matrix to this one
            @param other the other dense matrix
        */
        DenseMatrix& operator-=(const DenseMatrix& other);

        /*! @brief Multiply by a scalar */
        DenseMatrix& operator*=(double val);

        /*! @brief Divide by a scalar */
        DenseMatrix& operator/=(double val);

        /*! @brief Adds two matrices together A + B = C
            @param lhs the left hand side matrix A
            @param rhs the right hand side matrix B
            @retval The sum of the matrices C
        */
        friend DenseMatrix operator+(DenseMatrix lhs, const DenseMatrix& rhs);

        /*! @brief Subtract two matrices A - B = C
            @param lhs the left hand side matrix A
            @param rhs the right hand side matrix B
            @retval The difference of the matrices C
        */
        friend DenseMatrix operator-(DenseMatrix lhs, const DenseMatrix& rhs);

        /*! @brief Multiply a matrix by a scalar Aa = C
            @param lhs the left hand side matrix A
            @param val the scalar a
            @retval The multiplied matrix C
        */
        friend DenseMatrix operator*(DenseMatrix lhs, double val);

        /*! @brief Multiply a matrix by a scalar aA = C
            @param val the scalar a
            @param rhs the right hand side matrix A
            @retval The multiplied matrix C
        */
        friend DenseMatrix operator*(double val, DenseMatrix rhs);

        /*! @brief Divide all entries a matrix by a scalar A/a = C
            @param lhs the left hand side matrix A
            @param val the scalar a
            @retval The divided matrix C
        */
        friend DenseMatrix operator/(DenseMatrix lhs, double val);

        /*! @brief Divide a scalar by all entries of a matrix a/A = C
            @param val the scalar a
            @param rhs the right hand side matrix A
            @retval The divided matrix C
        */
        friend DenseMatrix operator/(double val, DenseMatrix rhs);

        /*! @brief Set all entries to a scalar value
            @param val the scalar a
        */
        DenseMatrix& operator=(double val);

        /*! @brief Check if the dense matrices are equal
            @param other the other DenseMatrix
            @retval true if the dense matrices are close enough to equal
        */
        bool operator==(const DenseMatrix& other) const;

        /*! @brief Get a single column view from the matrix,
                   Avoids the copy, but must own data
            @param col the column to get
            @param vect set this vect to the column values
        */
        VectorView<double> GetColView(int col);

        /*! @brief Get a single column view from the matrix,
                   Avoids the copy, but return immutable view
            @param col the column to get
            @param vect set this vect to the column values
        */
        const VectorView<double> GetColView(int col) const;

        /*! @brief Get a single column from the matrix
            @param col the column to get
            @param vect set this vect to the column values
        */
        void GetCol(int col, VectorView<double>& vect);

        /*! @brief Get a single column from the matrix
            @param col the column to get
            @param vect set this vect to the column values
        */
        template <typename T = double>
        void GetCol(int col, VectorView<T>& vect) const;

        /*! @brief Get a single row from the matrix
            @param row the row to get
            @param vect set this vect to the row values
        */
        template <typename T = double>
        void GetRow(int row, VectorView<T>& vect) const;

        /*! @brief Get a single column from the matrix
            @param col the column to get
            @retval vect the vect of column values
        */
        template <typename T = double>
        Vector<T> GetCol(int col) const;

        /*! @brief Get a single row from the matrix
            @param row the row to get
            @retval vect the vect of row values
        */
        template <typename T = double>
        Vector<T> GetRow(int row) const;

        /*! @brief Set a single column vector's values
            @param col the column to set
            @param vect the values to set
        */
        template <typename T = double>
        void SetCol(int col, const VectorView<T>& vect);

        /*! @brief Set a single row vector's values
            @param row the row to set
            @param vect the values to set
        */
        template <typename T = double>
        void SetRow(int row, const VectorView<T>& vect);

        /*! @brief Get a range of rows from the matrix
            @param start start of range, inclusive
            @param end end of range, inclusive
            @retval DenseMatrix the range of rows
        */
        DenseMatrix GetRow(int start, int end) const;

        /*! @brief Get a range of rows from the matrix
            @param start start of range, inclusive
            @param end end of range, inclusive
            @param dense dense matrix that will hold the range
        */
        void GetRow(int start, int end, DenseMatrix& dense) const;

        /*! @brief Get a selection of rows from the matrix
            @param rows rows to select
            @retval DenseMatrix the selection of rows
        */
        DenseMatrix GetRow(const std::vector<int>& rows) const;

        /*! @brief Get a selection of rows from the matrix
            @param rows rows to select
            @param dense dense matrix that will hold the selection
        */
        void GetRow(const std::vector<int>& rows, DenseMatrix& dense) const;

        /*! @brief Set a range of rows from the matrix
            @param start start of range, inclusive
            @param dense dense matrix that holds the range
        */
        void SetRow(int start, const DenseMatrix& dense);

        /*! @brief Get a range of columns from the matrix
            @param start start of range, inclusive
            @param end end of range, exclusive
            @retval DenseMatrix the range of columns
        */
        DenseMatrix GetCol(int start, int end) const;

        /*! @brief Get a range of columns from the matrix
            @param start start of range, inclusive
            @param end end of range, exclusive
            @param dense dense matrix that will hold the range
        */
        void GetCol(int start, int end, DenseMatrix& dense) const;

        /*! @brief Set a range of columns from the matrix
            @param start start of range, inclusive
            @param dense dense matrix that holds the range
        */
        void SetCol(int start, const DenseMatrix& dense);

        /*! @brief Get a contiguous submatrix for the range:
                 (start_i, start_j) to (end_i, end_j);
            @param start_i start of row range, inclusive
            @param start_j start of col range, inclusive
            @param end_i end of row range, exclusive
            @param end_j end of col range, exclusive
            @retval dense dense matrix that will hold the range
        */
        DenseMatrix GetSubMatrix(int start_i, int start_j, int end_i, int end_j) const;

        /*! @brief Get a contiguous submatrix for the range:
                 (start_i, start_j) to (end_i, end_j);
            @param start_i start of row range, inclusive
            @param start_j start of col range, inclusive
            @param end_i end of row range, exclusive
            @param end_j end of col range, exclusive
            @param dense dense matrix that will hold the range
        */
        void GetSubMatrix(int start_i, int start_j, int end_i, int end_j, DenseMatrix& dense) const;

        /*! @brief Get a submatrix with given indices
            @param rows row indices
            @param cols column indices
            @returns dense dense matrix that will hold the submatrix
        */
        DenseMatrix GetSubMatrix(const std::vector<int>& rows, const std::vector<int>& cols) const;

        /*! @brief Get a submatrix with given indices
            @param rows row indices
            @param cols column indices
            @param dense dense matrix that will hold the submatrix
        */
        void GetSubMatrix(const std::vector<int>& rows, const std::vector<int>& cols, DenseMatrix& dense) const;

        /*! @brief Set a contiguous submatrix for the range:
                 (start_i, start_j) to (end_i, end_j),
                 where end is determined by the input dense matrix
            @param start_i start of row range, inclusive
            @param start_j start of col range, inclusive
            @param dense dense matrix that holds the range
        */
        void SetSubMatrix(int start_i, int start_j, const DenseMatrix& dense);

        /*! @brief Set a contiguous submatrix for the range:
                 (start_i, start_j) to (end_i, end_j);
            @param start_i start of row range, inclusive
            @param start_j start of col range, inclusive
            @param end_i end of row range, exclusive
            @param end_j end of col range, exclusive
            @param dense dense matrix that holds the range
        */
        void SetSubMatrix(int start_i, int start_j, int end_i, int end_j, const DenseMatrix& dense);

        /*! @brief Set a transposed contiguous submatrix for the range:
                 (start_i, start_j) to (end_i, end_j),
                 where end is determined by the input dense matrix
            @param start_i start of row range, inclusive
            @param start_j start of col range, inclusive
            @param dense dense matrix that holds the range
        */
        void SetSubMatrixTranspose(int start_i, int start_j, const DenseMatrix& dense);

        /*! @brief Set a transposed contiguous submatrix for the range:
                 (start_i, start_j) to (end_i, end_j);
            @param start_i start of row range, inclusive
            @param start_j start of col range, inclusive
            @param end_i end of row range, exclusive
            @param end_j end of col range, exclusive
            @param dense dense matrix that holds the range
        */
        void SetSubMatrixTranspose(int start_i, int start_j, int end_i, int end_j, const DenseMatrix& dense);

        /*! @brief Add a non-contigious submatrix, given the rows and cols
            @param rows rows to add to
            @param cols cols to add to
            @param dense dense matrix that holds the submatrix
        */
        void AddSubMatrix(const std::vector<int>& rows, std::vector<int>& cols, const DenseMatrix& dense);

        /*! @brief Compute singular values and vectors A = U * S * VT
                   Where S is returned and A is replaced with VT
            @warning this replaces this matrix with U!
            @returns singular_values the computed singular_values
        */
        std::vector<double> SVD();

        /*! @brief Compute singular values and vectors A = U * S * VT
                   Where S is returned and A is replaced with U
            @param[out] VT DenseMatrix to hold the computed U
            @returns singular_values the computed singular_values
        */
        std::vector<double> SVD(DenseMatrix& U) const;

        /*! @brief Compute QR decomposition
            @warning this replaces this matrix with Q!
        */
        void QR();

        /*! @brief Compute QR decomposition
            @param Q Stores Q instead of overwriting
        */
        void QR(DenseMatrix& Q) const;

        /*! @brief Compute the inverse using LU factorization
            @warning this replaces this matrix with its inverse!
        */
        void Invert();

        /*! @brief Compute the inverse using LU factorization
            @param inv Stores inv instead of overwriting
        */
        void Invert(DenseMatrix& inv) const;

        /*! @brief Scale rows by given values
            @param values scale per row
        */
        template <typename T>
        void ScaleRows(const T& values);

        /*! @brief Scale cols by given values
            @param values scale per cols
        */
        template <typename T>
        void ScaleCols(const T& values);

        /*! @brief Scale rows by inverse of given values
            @param values scale per row
        */
        template <typename T>
        void InverseScaleRows(const T& values);

        /*! @brief Scale cols by inverse of given values
            @param values scale per cols
        */
        template <typename T>
        void InverseScaleCols(const T& values);

        /*! @brief Get Diagonal entries
            @returns diag diagonal entries
        */
        std::vector<double> GetDiag() const;

        /*! @brief Get Diagonal entries
            @param array to hold diag diagonal entries
        */
        void GetDiag(std::vector<double>& diag) const;

        /// Operator Requirement, calls the templated Mult
        void Mult(const VectorView<double>& input, VectorView<double> output) const override;
        /// Operator Requirement, calls the templated MultAT
        void MultAT(const VectorView<double>& input, VectorView<double> output) const override;

        using Operator::Mult;
        using Operator::MultAT;

    private:
        std::vector<double> data_;

        void dgemm(const DenseMatrix& input, DenseMatrix& output, bool AT, bool BT) const;

};

inline
double& DenseMatrix::operator()(int row, int col)
{
    assert(row >= 0);
    assert(col >= 0);

    assert(row < rows_);
    assert(col < cols_);

    return data_[row + (col * rows_)];
}

inline
const double& DenseMatrix::operator()(int row, int col) const
{
    assert(row >= 0);
    assert(col >= 0);

    assert(row < rows_);
    assert(col < cols_);

    return data_[row + (col * rows_)];
}

inline
double* DenseMatrix::GetData()
{
    return data_.data();
}

inline
const double* DenseMatrix::GetData() const
{
    return data_.data();
}

inline
double DenseMatrix::Sum() const
{
    assert(data_.size());

    double sum = std::accumulate(begin(data_), end(data_), 0.0);
    return sum;
}

inline
double DenseMatrix::Max() const
{
    assert(data_.size());

    return *std::max_element(begin(data_), end(data_));
}

inline
double DenseMatrix::Min() const
{
    assert(data_.size());

    return *std::min_element(begin(data_), end(data_));
}

template <typename T>
Vector<double> DenseMatrix::Mult(const VectorView<T>& input) const
{
    Vector<double> output(rows_);
    Mult(input, output);

    return output;
}

template <typename T, typename U>
void DenseMatrix::Mult(const VectorView<T>& input, VectorView<U> output) const
{
    assert(input.size() == cols_);
    assert(output.size() == rows_);

    output = 0;

    for (int j = 0; j < cols_; ++j)
    {
        for (int i = 0; i < rows_; ++i)
        {
            output[i] += (*this)(i, j) * input[j];
        }
    }
}

template <typename T>
Vector<double> DenseMatrix::MultAT(const VectorView<T>& input) const
{
    Vector<double> output(cols_);
    MultAT(input, output);

    return output;
}

template <typename T, typename U>
void DenseMatrix::MultAT(const VectorView<T>& input, VectorView<U> output) const
{
    assert(input.size() == rows_);
    assert(output.size() == cols_);

    for (int j = 0; j < cols_; ++j)
    {
        U val = 0;

        for (int i = 0; i < rows_; ++i)
        {
            val += (*this)(i, j) * input[i];
        }

        output[j] = val;
    }
}

inline
VectorView<double> DenseMatrix::GetColView(int col)
{
    assert(col >= 0 && col < cols_);

    return VectorView<double>(data_.data() + (col * rows_), rows_);
}

inline
const VectorView<double> DenseMatrix::GetColView(int col) const
{
    assert(col >= 0 && col < cols_);
    double* data = const_cast<double*>(data_.data());

    return VectorView<double>(data + (col * rows_), rows_);
}

inline
void DenseMatrix::GetCol(int col, VectorView<double>& vect)
{
    assert(col >= 0 && col < cols_);

    vect = VectorView<double>(data_.data() + (col * rows_), rows_);
}

template <typename T>
void DenseMatrix::GetCol(int col, VectorView<T>& vect) const
{
    assert(col >= 0 && col < cols_);
    assert(vect.size() == rows_);

    for (int i = 0; i < rows_; ++i)
    {
        vect[i] = (*this)(i, col);
    }
}

template <typename T>
void DenseMatrix::GetRow(int row, VectorView<T>& vect) const
{
    assert(row >= 0 && row < rows_);
    assert(vect.size() == cols_);

    for (int i = 0; i < cols_; ++i)
    {
        vect[i] = (*this)(row, i);
    }
}

template <typename T>
Vector<T> DenseMatrix::GetCol(int col) const
{
    Vector<T> vect(rows_);
    GetCol(col, vect);

    return vect;
}

template <typename T>
Vector<T> DenseMatrix::GetRow(int row) const
{
    Vector<T> vect(cols_);
    GetRow(row, vect);

    return vect;
}

template <typename T>
void DenseMatrix::SetCol(int col, const VectorView<T>& vect)
{
    assert(col >= 0 && col < cols_);
    assert(vect.size() == rows_);

    for (int i = 0; i < rows_; ++i)
    {
        (*this)(i, col) = vect[i];
    }
}

template <typename T>
void DenseMatrix::SetRow(int row, const VectorView<T>& vect)
{
    assert(row >= 0 && row < rows_);
    assert(vect.size() == cols_);

    for (int i = 0; i < cols_; ++i)
    {
        (*this)(row, i) = vect[i];
    }
}

template <typename T>
void DenseMatrix::ScaleRows(const T& values)
{
    for (int j = 0; j < cols_; ++j)
    {
        for (int i = 0; i < rows_; ++i)
        {
            (*this)(i, j) *= values[i];
        }
    }
}

template <typename T>
void DenseMatrix::ScaleCols(const T& values)
{
    for (int j = 0; j < cols_; ++j)
    {
        const double scale = values[j];

        for (int i = 0; i < rows_; ++i)
        {
            (*this)(i, j) *= scale;
        }
    }
}

template <typename T>
void DenseMatrix::InverseScaleRows(const T& values)
{
    for (int j = 0; j < cols_; ++j)
    {
        for (int i = 0; i < rows_; ++i)
        {
            assert(values[i] != 0.0);

            (*this)(i, j) /= values[i];
        }
    }
}

template <typename T>
void DenseMatrix::InverseScaleCols(const T& values)
{
    for (int j = 0; j < cols_; ++j)
    {
        const double scale = values[j];
        assert(scale != 0.0);

        for (int i = 0; i < rows_; ++i)
        {
            (*this)(i, j) /= scale;
        }
    }
}


// Utility Functions
DenseMatrix HStack(const std::vector<DenseMatrix>& dense);
void HStack(const std::vector<DenseMatrix>& dense, DenseMatrix& output);
//DenseMatrix VStack(const std::vector<DenseMatrix>& dense);

} //namespace linalgcpp

#endif // DENSEMATRIX_HPP__
