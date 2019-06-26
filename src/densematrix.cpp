#include "densematrix.hpp"

extern "C"
{
    void dgemm_(const char* transA, const char* transB,
                const int* m, const int* n, const int* k,
                const double* alpha, const double* A, const int* lda,
                const double* B, const int* ldb, const double* beta,
                double* C, const int* ldc);

    void dgesvd_(const char* jobu, const char* jobvt, const int* m, const int* n,
                 double* A, const int* lda, double* S, double* U, const int* ldu,
                 double* VT, const int* ldvt, double* work, const int* lwork, int* info);

    void dgeqp3_(const int* m, const int* n, double* A, const int* lda,
                 int* jpvt, double* tau, double* work, const int* lwork, int* info);

    void dorgqr_(const int* m, const int* n, const int* k, double* A,
                 const int* lda, double* tau, double* work, const int* lwork,
                 int* info);

    void dgetrf_(const int* m, const int* n, double* A, const int* lda,
                 int* ipiv, int* info);

    void dgetri_(const int* n, double* A, const int* lda, int* ipiv,
                 double* work, const int* lwork, int* info);
}

namespace linalgcpp
{

DenseMatrix::DenseMatrix()
{
}

DenseMatrix::DenseMatrix(int size)
    : DenseMatrix(size, size)
{
}

DenseMatrix::DenseMatrix(int rows, int cols)
    : DenseMatrix(rows, cols, std::vector<double>(rows * cols, 0.0))
{
}

DenseMatrix::DenseMatrix(int rows, int cols, std::vector<double> data)
    : Operator(rows, cols), data_(std::move(data))
{
    assert(rows >= 0);
    assert(cols >= 0);

    assert(static_cast<int>(data_.size()) == rows * cols);
}

DenseMatrix::DenseMatrix(const DenseMatrix& other) noexcept
    : Operator(other), data_(other.data_)
{
}

DenseMatrix::DenseMatrix(DenseMatrix&& other) noexcept
{
    swap(*this, other);
}

DenseMatrix& DenseMatrix::operator=(DenseMatrix other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(DenseMatrix& lhs, DenseMatrix& rhs) noexcept
{
    swap(static_cast<Operator&>(lhs), static_cast<Operator&>(rhs));

    std::swap(lhs.data_, rhs.data_);
}

void DenseMatrix::CopyData(std::vector<double>& data) const
{
    data.resize(data_.size());
    std::copy(std::begin(data_), std::end(data_), std::begin(data));
}

void DenseMatrix::SetSize(int size)
{
    SetSize(size, size);
}

void DenseMatrix::SetSize(int rows, int cols)
{
    assert(rows >= 0);
    assert(cols >= 0);

    rows_ = rows;
    cols_ = cols;
    data_.resize(rows * cols);
}

void DenseMatrix::SetSize(int size, double val)
{
    SetSize(size, size, val);
}

void DenseMatrix::SetSize(int rows, int cols, double val)
{
    assert(rows >= 0);
    assert(cols >= 0);

    rows_ = rows;
    cols_ = cols;
    data_.resize(rows * cols, val);
}


void DenseMatrix::Print(const std::string& label, std::ostream& out, int width, int precision) const
{
    std::stringstream ss;
    ss << label << "\n";

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < cols_; ++j)
        {
            ss << std::setw(width) << std::setprecision(precision)
                << std::fixed << (*this)(i, j);
            //<< std::defaultfloat << (*this)(i, j);
        }

        ss << "\n";
    }

    ss << "\n";

    out << ss.str();
}

DenseMatrix DenseMatrix::Transpose() const
{
    DenseMatrix transpose(cols_, rows_);

    Transpose(transpose);

    return transpose;
}

void DenseMatrix::Transpose(DenseMatrix& transpose) const
{
    transpose.SetSize(cols_, rows_);

    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < cols_; ++j)
        {
            transpose(j, i) = (*this)(i, j);
        }
    }
}

void DenseMatrix::Mult(const VectorView<double>& input, VectorView<double> output) const
{
    Mult<double, double>(input, output);
}

void DenseMatrix::MultAT(const VectorView<double>& input, VectorView<double> output) const
{
    MultAT<double, double>(input, output);
}

DenseMatrix DenseMatrix::Mult(const DenseMatrix& input) const
{
    DenseMatrix output(rows_, input.Cols());
    Mult(input, output);

    return output;
}

DenseMatrix DenseMatrix::MultAT(const DenseMatrix& input) const
{
    DenseMatrix output(cols_, input.Cols());
    MultAT(input, output);

    return output;
}

DenseMatrix DenseMatrix::MultBT(const DenseMatrix& input) const
{
    DenseMatrix output(rows_, input.Rows());
    MultBT(input, output);

    return output;
}

DenseMatrix DenseMatrix::MultABT(const DenseMatrix& input) const
{
    DenseMatrix output(cols_, input.Rows());
    MultABT(input, output);

    return output;
}

void DenseMatrix::Mult(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(cols_ == input.Rows());

    bool AT = false;
    bool BT = false;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::MultAT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(rows_ == input.Rows());

    bool AT = true;
    bool BT = false;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::MultBT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(cols_ == input.Cols());

    bool AT = false;
    bool BT = true;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::MultABT(const DenseMatrix& input, DenseMatrix& output) const
{
    assert(rows_ == input.Cols());

    bool AT = true;
    bool BT = true;
    dgemm(input, output, AT, BT);
}

void DenseMatrix::dgemm(const DenseMatrix& input, DenseMatrix& output, bool AT, bool BT) const
{
    char transA = AT ? 'T' : 'N';
    char transB = BT ? 'T' : 'N';
    int m = AT ? cols_ : rows_;
    int n = BT ? input.Rows() : input.Cols();
    int k = AT ? rows_ : cols_;

    double alpha = 1.0;
    const double* A = data_.data();
    int lda = rows_;
    const double* B = input.data_.data();
    int ldb = input.Rows();
    double beta = 0.0;

    output.SetSize(m, n);
    double* c = output.data_.data();
    int ldc = output.Rows();


    dgemm_(&transA, &transB, &m, &n, &k,
           &alpha, A, &lda, B, &ldb,
           &beta, c, &ldc);
}

DenseMatrix& DenseMatrix::operator-=(const DenseMatrix& other)
{
    assert(rows_ == other.Rows());
    assert(cols_ == other.Cols());

    const int nnz = rows_ * cols_;

    for (int i = 0; i < nnz; ++i)
    {
        data_[i] -= other.data_[i];
    }

    return *this;
}

DenseMatrix& DenseMatrix::operator+=(const DenseMatrix& other)
{
    assert(rows_ == other.Rows());
    assert(cols_ == other.Cols());

    const int nnz = rows_ * cols_;

    for (int i = 0; i < nnz; ++i)
    {
        data_[i] += other.data_[i];
    }

    return *this;
}

DenseMatrix operator+(DenseMatrix lhs, const DenseMatrix& rhs)
{
    return lhs += rhs;
}

DenseMatrix operator-(DenseMatrix lhs, const DenseMatrix& rhs)
{
    return lhs -= rhs;
}

DenseMatrix& DenseMatrix::operator*=(double val)
{
    for (auto& i : data_)
    {
        i *= val;
    }

    return *this;
}

DenseMatrix operator*(DenseMatrix lhs, double val)
{
    return lhs *= val;
}

DenseMatrix operator*(double val, DenseMatrix rhs)
{
    return rhs *= val;
}

DenseMatrix& DenseMatrix::operator/=(double val)
{
    assert(val != 0);

    for (auto& i : data_)
    {
        i /= val;
    }

    return *this;
}

DenseMatrix operator/(DenseMatrix lhs, double val)
{
    return lhs /= val;
}

DenseMatrix& DenseMatrix::operator=(double val)
{
    std::fill(begin(data_), end(data_), val);

    return *this;
}

bool DenseMatrix::operator==(const DenseMatrix& other) const
{
    if (other.Rows() != rows_ || other.Cols() != cols_)
    {
        return false;
    }

    constexpr double tol = 1e-12;

    for (int j = 0; j < cols_; ++j)
    {
        for (int i = 0; i < rows_; ++i)
        {
            if (std::fabs((*this)(i, j) - other(i, j)) > tol)
            {
                return false;
            }
        }
    }

    return true;
}

DenseMatrix DenseMatrix::GetRow(int start, int end) const
{
    const int num_rows = end - start;
    DenseMatrix dense(num_rows, cols_);

    GetRow(start, end, dense);

    return dense;
}

void DenseMatrix::GetRow(int start, int end, DenseMatrix& dense) const
{
    GetSubMatrix(start, 0, end, cols_, dense);
}

DenseMatrix DenseMatrix::GetRow(const std::vector<int>& rows) const
{
    DenseMatrix dense(rows.size(), Cols());

    GetRow(rows, dense);

    return dense;
}

void DenseMatrix::GetRow(const std::vector<int>& rows, DenseMatrix& dense) const
{
    assert(dense.Cols() == Cols());
    assert(dense.Rows() == static_cast<int>(rows.size()));

    const int num_rows = rows.size();
    const int num_cols = Cols();

    for (int i = 0; i < num_rows; ++i)
    {
        const int row = rows[i];

        for (int j = 0; j < num_cols; ++j)
        {
            dense(i, j) = (*this)(row, j);
        }
    }
}

void DenseMatrix::SetRow(int start, const DenseMatrix& dense)
{
    const int end = start + dense.Rows();
    SetSubMatrix(start, 0, end, cols_, dense);
}

DenseMatrix DenseMatrix::GetCol(int start, int end) const
{
    const int num_cols = end - start;
    DenseMatrix dense(rows_, num_cols);

    GetCol(start, end, dense);

    return dense;
}

void DenseMatrix::GetCol(int start, int end, DenseMatrix& dense) const
{
    GetSubMatrix(0, start, rows_, end, dense);
}

void DenseMatrix::SetCol(int start, const DenseMatrix& dense)
{
    const int end = start + dense.Cols();
    SetSubMatrix(0, start, rows_, end, dense);
}

DenseMatrix DenseMatrix::GetSubMatrix(int start_i, int start_j, int end_i, int end_j) const
{
    const int num_rows = end_i - start_i;
    const int num_cols = end_j - start_j;

    DenseMatrix dense(num_rows, num_cols);
    GetSubMatrix(start_i, start_j, end_i, end_j, dense);

    return dense;
}

void DenseMatrix::GetSubMatrix(int start_i, int start_j, int end_i, int end_j, DenseMatrix& dense) const
{
    assert(start_i >= 0 && start_i < rows_);
    assert(start_j >= 0 && start_j < cols_);
    assert(end_i >= 0 && end_i <= rows_);
    assert(end_j >= 0 && end_j <= cols_);
    assert(end_i >= start_i && end_j >= start_j);

    const int num_rows = end_i - start_i;
    const int num_cols = end_j - start_j;

    dense.SetSize(num_rows, num_cols);

    for (int j = 0; j < num_cols; ++j)
    {
        for (int i = 0; i < num_rows; ++i)
        {
            dense(i, j) = (*this)(i + start_i, j + start_j);
        }
    }
}

DenseMatrix DenseMatrix::GetSubMatrix(const std::vector<int>& rows, const std::vector<int>& cols) const
{
    DenseMatrix out;
    GetSubMatrix(rows, cols, out);

    return out;
}

void DenseMatrix::GetSubMatrix(const std::vector<int>& rows, const std::vector<int>& cols, DenseMatrix& dense) const
{
    int num_rows = rows.size();
    int num_cols = cols.size();

    dense.SetSize(num_rows, num_cols);
    dense = 0.0;

    for (int j = 0; j < num_cols; ++j)
    {
        int col = cols[j];

        for (int i = 0; i < num_rows; ++i)
        {
            int row = rows[i];

            dense(i, j) = (*this)(row, col);
        }
    }
}

void DenseMatrix::SetSubMatrix(int start_i, int start_j, const DenseMatrix& dense)
{
    SetSubMatrix(start_i, start_j, start_i + dense.Rows(), start_j + dense.Cols(), dense);
}

void DenseMatrix::SetSubMatrix(int start_i, int start_j, int end_i, int end_j, const DenseMatrix& dense)
{
    assert(start_i >= 0 && start_i < rows_);
    assert(start_j >= 0 && start_j < cols_);
    assert(end_i >= 0 && end_i <= rows_);
    assert(end_j >= 0 && end_j <= cols_);
    assert(end_i >= start_i && end_j >= start_j);

    const int num_rows = end_i - start_i;
    const int num_cols = end_j - start_j;

    for (int j = 0; j < num_cols; ++j)
    {
        for (int i = 0; i < num_rows; ++i)
        {
            (*this)(i + start_i, j + start_j) = dense(i, j);
        }
    }
}

void DenseMatrix::SetSubMatrixTranspose(int start_i, int start_j, const DenseMatrix& dense)
{
    SetSubMatrixTranspose(start_i, start_j, start_i + dense.Cols(), start_j + dense.Rows(), dense);
}

void DenseMatrix::SetSubMatrixTranspose(int start_i, int start_j, int end_i, int end_j, const DenseMatrix& dense)
{
    assert(start_i >= 0 && start_i < rows_);
    assert(start_j >= 0 && start_j < cols_);
    assert(end_i >= 0 && end_i <= rows_);
    assert(end_j >= 0 && end_j <= cols_);
    assert(end_i >= start_i && end_j >= start_j);

    const int num_rows = end_i - start_i;
    const int num_cols = end_j - start_j;

    for (int j = 0; j < num_cols; ++j)
    {
        for (int i = 0; i < num_rows; ++i)
        {
            (*this)(i + start_i, j + start_j) = dense(j, i);
        }
    }
}

void DenseMatrix::AddSubMatrix(const std::vector<int>& rows, std::vector<int>& cols, const DenseMatrix& dense)
{
    assert(static_cast<int>(rows.size()) == dense.Rows());
    assert(static_cast<int>(cols.size()) == dense.Cols());

    int num_rows = dense.Rows();
    int num_cols = dense.Cols();

    for (int j = 0; j < num_cols; ++j)
    {
        for (int i = 0; i < num_rows; ++i)
        {
            (*this)(rows[i], cols[j]) += dense(i, j);
        }
    }
}

std::vector<double> DenseMatrix::SVD(DenseMatrix& U) const
{
    U = *this;

    return U.SVD();
}

std::vector<double> DenseMatrix::SVD()
{
    const int rows = Rows();
    const int cols = Cols();

    std::vector<double> singular_values(std::min(rows, cols));

    if (rows == 0 || cols == 0)
    {
        return singular_values;
    }

    const char* jobu = "O";
    const char* jobvt = "N";
    const int* m = &rows;
    const int* n = &cols;
    double* A = data_.data();
    const int* lda = &rows;
    double* S = singular_values.data();
    double* U = nullptr;
    const int* ldu = &rows;
    double* VT = nullptr;
    const int* ldvt = &cols;
    int info;

    int lwork = -1;
    double qwork;

    dgesvd_(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
            &qwork, &lwork, &info);

    lwork = static_cast<int>(qwork);

    std::vector<double> work(lwork);

    dgesvd_(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt,
            work.data(), &lwork, &info);

    assert(info == 0);

    return singular_values;
}

/*
    void dgeqp3_(const int* m, const int* n, double* A, const int* lda,
                 int* jpvt, double* tau, double* work, const int* lwork, int* info);
    void dorgqr_(const int* m, const int* n, const int* k, double* A,
                 const int* lda, double* tau, double* work, const int* lwork,
                 int* info);
*/

void DenseMatrix::QR(DenseMatrix& Q) const
{
    Q = *this;
    Q.QR();
}

void DenseMatrix::QR()
{
    const int rows = Rows();
    const int cols = Cols();

    if (rows == 0 || cols == 0)
    {
        return;
    }

    const int mn_min = std::min(rows, cols);

    const int* m = &rows;
    const int* n = &cols;
    double* A = data_.data();
    const int* lda = &rows;
    int info;

    int lwork = -1;
    double qwork;

    dgeqp3_(m, n, A, lda, nullptr, nullptr, &qwork, &lwork, &info);

    lwork = static_cast<int>(qwork);

    std::vector<int> jpvt(mn_min, 0);
    std::vector<double> tau(mn_min);
    std::vector<double> work(lwork);

    dgeqp3_(m, n, A, lda, jpvt.data(), tau.data(), work.data(), &lwork, &info);

    const int* k = &mn_min;

    lwork = -1;
    dorgqr_(m, n, k, A, lda, tau.data(), &qwork, &lwork, &info);
    lwork = static_cast<int>(qwork);

    work.resize(lwork);

    dorgqr_(m, n, k, A, lda, tau.data(), work.data(), &lwork, &info);

    assert(info == 0);
}

void DenseMatrix::Invert(DenseMatrix& inv) const
{
    inv = *this;
    inv.Invert();
}

void DenseMatrix::Invert()
{
    const int rows = Rows();
    const int cols = Cols();
    assert(rows == cols);

    if (rows == 0 || cols == 0)
    {
        return;
    }

    const int mn_min = std::min(rows, cols);

    const int* m = &rows;
    const int* n = &cols;
    double* A = data_.data();
    const int* lda = &rows;
    int info;

    std::vector<int> ipiv(mn_min, 0);

    // Factor
    dgetrf_(m, n, A, lda, ipiv.data(), &info);

    // Invert
    int lwork = -1;
    double qwork;

    dgetri_(n, A, lda, ipiv.data(), &qwork, &lwork, &info);

    lwork = static_cast<int>(qwork);
    std::vector<double> work(lwork);

    dgetri_(n, A, lda, ipiv.data(), work.data(), &lwork, &info);

    assert(info == 0);
}

std::vector<double> DenseMatrix::GetDiag() const
{
    assert(rows_ == cols_);

    std::vector<double> diag(rows_);

    GetDiag(diag);

    return diag;
}

void DenseMatrix::GetDiag(std::vector<double>& diag) const
{
    assert(rows_ == cols_);
    assert(static_cast<int>(diag.size()) == rows_);

    for (int i = 0; i < rows_; ++i)
    {
        diag[i] = (*this)(i, i);
    }
}

DenseMatrix HStack(const std::vector<DenseMatrix>& dense)
{
    DenseMatrix output;
    HStack(dense, output);

    return output;
}

void HStack(const std::vector<DenseMatrix>& dense, DenseMatrix& output)
{
    if (dense.size() == 0)
    {
        return;
    }

    int rows = dense[0].Rows();

    for (auto& mat : dense)
    {
        assert(mat.Rows() == rows);
    }

    int cols = SumCols(dense);

    output.SetSize(rows, cols);

    int counter = 0;

    for (auto& mat : dense)
    {
        if (mat.Cols() > 0)
        {
            output.SetCol(counter, mat);
            counter += mat.Cols();
        }
    }
}

} // namespace linalgcpp

