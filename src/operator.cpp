/* @file */

#include "operator.hpp"

namespace linalgcpp
{

Operator::Operator()
    : Operator(0)
{
}

Operator::Operator(int size)
    : Operator(size, size)
{
}

Operator::Operator(int rows, int cols)
    : rows_(rows), cols_(cols)
{
    assert(rows_ >= 0);
    assert(cols_ >= 0);
}

Operator::Operator(const Operator& other) noexcept
    : rows_(other.rows_), cols_(other.cols_)
{
    assert(rows_ >= 0);
    assert(cols_ >= 0);
}

Operator::Operator(Operator&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_)
{
    assert(rows_ >= 0);
    assert(cols_ >= 0);
}
Operator& Operator::operator=(Operator&& other) noexcept
{
    swap(*this, other);
    return *this;
}

int Operator::Rows() const
{
    return rows_;
}

int Operator::Cols() const
{
    return cols_;
}

void swap(Operator& lhs, Operator& rhs) noexcept
{
    std::swap(lhs.rows_, rhs.rows_);
    std::swap(lhs.cols_, rhs.cols_);
}

void Operator::Mult(const VectorView<double>& input, Vector<double>& output) const
{
    output.SetSize(Rows());

    Mult(input, static_cast<VectorView<double>&>(output));
}

Vector<double> Operator::Mult(const VectorView<double>& input) const
{
    Vector<double> output(Rows());
    Mult(input, output);

    return output;
}

void Operator::MultAT(const VectorView<double>& input, Vector<double>& output) const
{
    output.SetSize(Cols());

    MultAT(input, static_cast<VectorView<double>&>(output));
}

Vector<double> Operator::MultAT(const VectorView<double>& input) const
{
    Vector<double> output(Cols());
    MultAT(input, output);

    return output;
}

double Operator::InnerProduct(const VectorView<double>& x, const VectorView<double>& y) const
{
    return y.Mult(Mult(x));
}

} // namespace linalgcpp
