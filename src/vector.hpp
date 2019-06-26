/*! @file */

#ifndef VECTOR_HPP__
#define VECTOR_HPP__

#include <memory>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <assert.h>

#include "vectorview.hpp"

namespace linalgcpp
{

template <typename T = double>
class Vector : public VectorView<T>
{
    public:
        /*! @brief Default Constructor of zero size */
        Vector() = default;

        /*! @brief Constructor of setting the size
            @param size the length of the vector
        */
        explicit Vector(int size);

        /*! @brief Constructor of setting the size and intial values
            @param size the length of the vector
            @param val the initial value to set all entries to
        */
        Vector(int size, T val);

        /*! @brief Copy Constructor given data and size
            @param data vector data
            @param size the length of the vector

            @note this is not a view, copies data
        */
        Vector(const T* data, int size);

        /*! @brief Constructor from view
            @note deep copy
         * */
        explicit Vector(const VectorView<T>& vect);

        /*! @brief Constructor from an std::vector*/
        explicit Vector(std::vector<T> vect);

        /*! @brief Copy Constructor */
        Vector(const Vector& vect) noexcept;

        /*! @brief Move constructor
            @param vect the vector to move
        */
        Vector(Vector&& vect) noexcept;

        /*! @brief Destructor
        */
        ~Vector() noexcept = default;

        /*! @brief Access data vector */
        const std::vector<T>& data() const { return data_; }

        /*! @brief Set size of vector
            @param size size to set

            @note entries not intialized
        */
        void SetSize(int size);

        /*! @brief Set size of vector and initialization value
            @param size size to set
            @param val value of any new entries
        */
        void SetSize(int size, T val);

        /*! @brief Sets this vector equal to another
            @param vect the vector to copy
        */

        Vector& operator=(const Vector<T>& vect) noexcept;

        /*! @brief Sets this vector equal to another
            @param vect the vector to copy
        */
        Vector& operator=(const VectorView<T>& vect) noexcept;

        /*! @brief Sets this vector equal to another
            @param vect the vector to copy
        */
        Vector& operator=(Vector&& vect) noexcept;

        /*! @brief Swap two vectors
            @param lhs left hand side vector
            @param rhs right hand side vector
        */
        template <typename U>
        friend void swap(Vector<U>& lhs, Vector<U>& rhs) noexcept;

        using VectorView<T>::operator=;
        virtual T Mult(const VectorView<T>& vect) const override;

    private:
        std::vector<T> data_;
};

template <typename T>
T Vector<T>::Mult(const VectorView<T>& vect) const
{
    return VectorView<T>::Mult(vect);
}

template <typename T>
Vector<T>::Vector(int size)
{
    data_.resize(size);

    VectorView<T>::SetData(data_.data(), data_.size());
}

template <typename T>
Vector<T>::Vector(int size, T val)
{
    data_.resize(size, val);

    VectorView<T>::SetData(data_.data(), data_.size());
}

template <typename T>
Vector<T>::Vector(const T* data, int size)
    : data_(data, data + size)
{
    assert(size >= 0);
    assert(size == 0 || data);

    VectorView<T>::SetData(data_.data(), data_.size());
}

template <typename T>
Vector<T>::Vector(const VectorView<T>& vect)
{
    data_.resize(vect.size());
    std::copy(std::begin(vect), std::end(vect), std::begin(data_));

    VectorView<T>::SetData(data_.data(), data_.size());
}

template <typename T>
Vector<T>::Vector(std::vector<T> vect)
{
    std::swap(vect, data_);

    VectorView<T>::SetData(data_.data(), data_.size());
}

template <typename T>
Vector<T>::Vector(const Vector<T>& vect) noexcept
    : data_(vect.data_)
{
    VectorView<T>::SetData(data_.data(), data_.size());
}

template <typename T>
Vector<T>::Vector(Vector<T>&& vect) noexcept
{
    swap(*this, vect);

    VectorView<T>::SetData(data_.data(), data_.size());
}

template <typename T>
void Vector<T>::SetSize(int size)
{
    if (size != VectorView<T>::size())
    {
        assert(size >= 0);

        data_.resize(size);

        VectorView<T>::SetData(data_.data(), data_.size());
    }
}

template <typename T>
void Vector<T>::SetSize(int size, T val)
{
    if (size != VectorView<T>::size())
    {
        assert(size >= 0);

        data_.resize(size, val);

        VectorView<T>::SetData(data_.data(), data_.size());
    }
}

template <typename T>
Vector<T>& Vector<T>::operator=(const Vector<T>& vect) noexcept
{
    data_.resize(vect.size());
    std::copy(std::begin(vect.data_), std::end(vect.data_), std::begin(data_));

    VectorView<T>::SetData(data_.data(), data_.size());

    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator=(const VectorView<T>& vect) noexcept
{
    data_.resize(vect.size());
    std::copy(std::begin(vect), std::end(vect), std::begin(data_));

    VectorView<T>::SetData(data_.data(), data_.size());

    return *this;
}

template <typename T>
Vector<T>& Vector<T>::operator=(Vector<T>&& vect) noexcept
{
    swap(*this, vect);

    VectorView<T>::SetData(data_.data(), data_.size());

    return *this;
}

template <typename U>
void swap(Vector<U>& lhs, Vector<U>& rhs) noexcept
{
    swap(static_cast<VectorView<U>&>(lhs), static_cast<VectorView<U>&>(rhs));
    std::swap(lhs.data_, rhs.data_);
}

/*! @brief Multiply a vector by a scalar into a new vector
    @param vect vector to multiply
    @param val value to scale by
    @retval the vector multiplied by the scalar
*/
template <typename T>
Vector<T> operator*(Vector<T> vect, T val)
{
    vect *= val;
    return vect;
}

/*! @brief Multiply a vector by a scalar into a new vector
    @param vect vector to multiply
    @param val value to scale by
    @retval the vector multiplied by the scalar
*/
template <typename T>
Vector<T> operator*(T val, Vector<T> vect)
{
    vect *= val;
    return vect;
}

/*! @brief Divide a vector by a scalar into a new vector
    @param vect vector to divide
    @param val value to scale by
    @retval the vector divided by the scalar
*/
template <typename T>
Vector<T> operator/(Vector<T> vect, T val)
{
    vect /= val;
    return vect;
}

/*! @brief Divide a scalar by vector entries into a new vector
    @param vect vector to multiply
    @param val value to scale by
    @retval the vector multiplied by the scalar
*/
template <typename T>
Vector<T> operator/(T val, Vector<T> vect)
{
    for (T& i : vect)
    {
        i = val / i;
    }

    return vect;
}

/*! @brief Add two vectors into a new vector z = x + y
    @param lhs left hand side vector x
    @param rhs right hand side vector y
    @retval the sum of the two vectors
*/
template <typename T>
Vector<T> operator+(Vector<T> lhs, const Vector<T>& rhs)
{
    lhs += rhs;
    return lhs;
}

/*! @brief Subtract two vectors into a new vector z = x - y
    @param lhs left hand side vector x
    @param rhs right hand side vector y
    @retval the difference of the two vectors
*/
template <typename T>
Vector<T> operator-(Vector<T> lhs, const Vector<T>& rhs)
{
    lhs -= rhs;
    return lhs;
}

} // namespace linalgcpp

#endif // VECTOR_HPP__
