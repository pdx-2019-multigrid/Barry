/*! @file */

#ifndef OPERATOR_HPP__
#define OPERATOR_HPP__

#include <memory>
#include <vector>
#include <numeric>
#include <assert.h>

#include "vector.hpp"

namespace linalgcpp
{

/*! @brief Abstract operator class that has a size
          and can apply its action to a vector
*/
class Operator
{
    public:
        /*! @brief Zero Size Constructor */
        Operator();

        /*! @brief Square Constructor */
        Operator(int size);

        /*! @brief Rectangle Constructor */
        Operator(int rows, int cols);

        /*! @brief Copy Constructor */
        Operator(const Operator&) noexcept;

        /*! @brief Move Constructor */
        Operator(Operator&&) noexcept;

        /*! @brief Assignement Operator */
        Operator& operator=(Operator&&) noexcept;

        /*! @brief Destructor */
        virtual ~Operator() noexcept = default;

        /*! @brief The number of rows in this operator
            @retval the number of rows
        */
        virtual int Rows() const;

        /*! @brief The number of columns in this operator
            @retval the number of columns
        */
        virtual int Cols() const;

        /*! @brief Swap two operator
            @param lhs left hand side operator
            @param rhs right hand side operator
        */
        friend void swap(Operator& lhs, Operator& rhs) noexcept;

        /*! @brief Apply the action to a vector: Ax = y
            @param input the input vector x
            @param output the output vector y
        */
        virtual void Mult(const VectorView<double>& input, VectorView<double> output) const = 0;

        /*! @brief Apply the action to a vector: Ax = y
            @param input the input vector x
            @param output the output vector y, automatically resized
        */
        virtual void Mult(const VectorView<double>& input, Vector<double>& output) const;

        /*! @brief Apply the action to a vector: Ax = y
            @param input the input vector x
            @retval output the output vector y
        */
        virtual Vector<double> Mult(const VectorView<double>& input) const;

        /*! @brief Apply the transpose action to a vector: A^T x = y
            @param input the input vector x
            @param output the output vector y
        */
        virtual void MultAT(const VectorView<double>& input, VectorView<double> output) const;

        /*! @brief Apply the transpose action to a vector: A^T x = y
            @param input the input vector x
            @param output the output vector y, automatically resized
        */
        virtual void MultAT(const VectorView<double>& input, Vector<double>& output) const;

        /*! @brief Apply the transpose action to a vector: A^T x = y
            @param input the input vector x
            @retval output the output vector y
        */
        virtual Vector<double> MultAT(const VectorView<double>& input) const;

        /*! @brief Computes the inner produt y^T A x
            @param x the input vector x
            @param y the input vector y
            @retval double the A inner product
        */
        double InnerProduct(const VectorView<double>& x, const VectorView<double>& y) const;
    protected:
        int rows_;
        int cols_;
};

inline
void Operator::MultAT(const VectorView<double>& input, VectorView<double> output) const
{
    throw std::runtime_error("The operator MultAT not defined!\n");
}

template <typename T>
inline
int SumCols(const std::vector<T>& ops)
{
    int total = 0;

    for (const auto& op : ops)
    {
        total += op.Cols();
    }

    return total;
}

template <typename T>
inline
int SumRows(const std::vector<T>& ops)
{
    int total = 0;

    for (const auto& op : ops)
    {
        total += op.Rows();
    }

    return total;
}

}
#endif // OPERATOR_HPP__
