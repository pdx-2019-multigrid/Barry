/*! @file */

#ifndef VECTORVIEW_HPP__
#define VECTORVIEW_HPP__

#include <memory>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <assert.h>

namespace linalgcpp
{

template <typename T>
class Vector;

/*! @brief Vector view of data and size

    @note Views are only modifiable if you own the view
          that you plan to change it to.

          If you want view A to be equal to view B,
          you most own both A and B. Otherwise,
          it is trivial to subvert const restrictions.
          I'm not sure if this is good way or not to
          deal w/ this.
*/
template <typename T = double>
class VectorView
{
    public:
        /*! @brief Default Constructor of zero size */
        VectorView();

        /*! @brief Constructor with data */
        VectorView(T* data, int size);

        /*! @brief Copy with vector */
        VectorView(std::vector<T>& vect) noexcept;

        /*! @brief Copy Constructor */
        VectorView(VectorView& vect) noexcept;

        /*! @brief Move Constructor */
        VectorView(VectorView&& vect) noexcept;

        /*! @brief Assignment Operator */
        VectorView& operator=(VectorView& vect) noexcept;

        /*! @brief Assignment Operator */
        VectorView& operator=(VectorView&& vect) noexcept;

        /*! @brief Assignment Operator, hard copy*/
        VectorView& operator=(const VectorView& vect) noexcept;

        /*! @brief Destructor */
        ~VectorView() noexcept = default;

        /*! @brief Swap two vectors
            @param lhs left hand side vector
            @param rhs right hand side vector
        */
        template <typename U>
        friend void swap(VectorView<U>& lhs, VectorView<U>& rhs) noexcept;

        /*! @brief STL like begin. Points to start of data
            @retval pointer to the start of data
        */
        virtual T* begin();

        /*! @brief STL like end. Points to the end of data
            @retval pointer to the end of data
        */
        virtual T* end();

        /*! @brief STL like const begin. Points to start of data
            @retval const pointer to the start of data
        */
        virtual const T* begin() const;

        /*! @brief STL like const end. Points to the end of data
            @retval const pointer to the end of data
        */
        virtual const T* end() const;

        /*! @brief Index operator
            @param i index into vector
            @retval reference to value at index i
        */
        virtual T& operator[](int i);

        /*! @brief Const index operator
            @param i index into vector
            @retval const reference to value at index i
        */
        virtual const T& operator[](int i) const;

        /*! @brief Get the length of the vector
            @retval the length of the vector
        */
        virtual int size() const;

        /*! @brief Sets all entries to a scalar value
            @param val the value to set all entries to
        */
        VectorView& operator=(T val);

        /*! @brief Get subvector
            @param indices indices of values to get
            @param vect Vector to hold subvector
        */
        void GetSubVector(const std::vector<int>& indices, Vector<T>& vect) const;

        /*! @brief Get subvector
            @param indices indices of values to get
            @returns vect Vector to hold subvector
        */
        Vector<T> GetSubVector(const std::vector<int>& indices) const;

        /*! @brief Print the vector entries
            @param label the label to print before the list of entries
            @param out stream to print to
        */
        virtual void Print(const std::string& label = "", std::ostream& out = std::cout) const;

        /*! @brief Inner product of two vectors
            @param vect other vector
        */
        virtual T Mult(const VectorView<T>& vect) const;

        /*! @brief Add (alpha * vect) to this vector
            @param alpha scale of rhs
            @param vect vector to add
        */
        VectorView<T>& Add(double alpha, const VectorView<T>& vect);

        /*! @brief Add vect to this vector
            @param vect vector to add
        */
        VectorView<T>& Add(const VectorView<T>& vect);

        /*! @brief Subtract (alpha * vect) from this vector
            @param alpha scale of rhs
            @param vect vector to subtract
        */
        VectorView<T>& Sub(double alpha, const VectorView<T>& vect);

        /*! @brief Subtract vect from this vector
            @param vect vector to subtract
        */
        VectorView<T>& Sub(const VectorView<T>& vect);

        /*! @brief Set this vector to (alpha * vect)
            @param alpha scale of rhs
            @param vect vector to set
        */
        VectorView<T>& Set(double alpha, const VectorView<T>& vect);

        /*! @brief Set this vector to vect
            @param vect vector to set
        */
        VectorView<T>& Set(const VectorView<T>& vect);

        /*! @brief Compute the L2 norm of the vector
            @retval the L2 norm
        */
        virtual double L2Norm() const;

        /*! @brief Sum all elements
            @retval sum sum of all elements
        */
        T Sum() const;

        /*! @brief Randomize the entries in a integer vector
            @param lo lower range limit
            @param hi upper range limit
            @param seed seed to rng, if positive
        */
        virtual void Randomize(int lo = 0, int hi = 1, int seed = -1);

        /*! @brief Normalize a vector */
        virtual void Normalize();

    protected:
        void SetData(T* data, int size);

    private:
        T* data_;
        int size_;
};

template <typename T>
VectorView<T>::VectorView()
    : data_(nullptr), size_(0)
{

}

template <typename T>
VectorView<T>::VectorView(T* data, int size)
    : data_(data), size_(size)
{
    assert(size_ >= 0);
    assert(size_ == 0 || data_);
}

template <typename T>
VectorView<T>::VectorView(VectorView<T>& vect) noexcept
    : data_(vect.data_), size_(vect.size_)
{
    assert(size_ >= 0);
    assert(size_ == 0 || data_);
}

template <typename T>
VectorView<T>::VectorView(std::vector<T>& vect) noexcept
    : data_(vect.data()), size_(vect.size())
{
    assert(size_ >= 0);
    assert(size_ == 0 || data_);
}

template <typename T>
void VectorView<T>::SetData(T* data, int size)
{
    assert(size >= 0);
    assert(size == 0 || data);

    data_ = data;
    size_ = size;
}

template <typename T>
VectorView<T>::VectorView(VectorView<T>&& vect) noexcept
    : data_(vect.data_), size_(vect.size_)
{
    assert(size_ >= 0);
    assert(size_ == 0 || data_);
}

template <typename T>
VectorView<T>& VectorView<T>::operator=(VectorView<T>& vect) noexcept
{
    assert(vect.size_ == size_);

    std::copy(std::begin(vect), std::end(vect), std::begin(*this));

    //data_ = vect.data_;
    //size_ = vect.size_;

    return *this;
}

template <typename T>
VectorView<T>& VectorView<T>::operator=(VectorView<T>&& vect) noexcept
{
    assert(vect.size_ == size_);
    std::copy(std::begin(vect), std::end(vect), std::begin(*this));

    //data_ = vect.data_;
    //size_ = vect.size_;

    return *this;
}

template <typename T>
VectorView<T>& VectorView<T>::operator=(const VectorView<T>& vect) noexcept
{
    assert(vect.size_ == size_);

    std::copy(std::begin(vect), std::end(vect), std::begin(*this));

    return *this;
}

template <typename U>
void swap(VectorView<U>& lhs, VectorView<U>& rhs) noexcept
{
    std::swap(lhs.data_, rhs.data_);
    std::swap(lhs.size_, rhs.size_);
}

template <typename T>
VectorView<T>& VectorView<T>::operator=(T val)
{
    std::fill(std::begin(*this), std::end(*this), val);

    return *this;
}

template <typename T>
T* VectorView<T>::begin()
{
    return data_;
}

template <typename T>
T* VectorView<T>::end()
{
    return data_ + size_;
}

template <typename T>
const T* VectorView<T>::begin() const
{
    return data_;
}

template <typename T>
const T* VectorView<T>::end() const
{
    return data_ + size_;
}

template <typename T>
T& VectorView<T>::operator[](int i)
{
    assert(i < size_);

    return data_[i];
}

template <typename T>
const T& VectorView<T>::operator[](int i) const
{
    assert(i < size_);

    return data_[i];
}

template <typename T>
int VectorView<T>::size() const
{
    return size_;
}

template <typename T>
Vector<T> VectorView<T>::GetSubVector(const std::vector<int>& indices) const
{
    Vector<T> vect(indices.size());

    GetSubVector(indices, vect);

    return vect;
}

template <typename T>
void VectorView<T>::GetSubVector(const std::vector<int>& indices, Vector<T>& vect) const
{
    int size = indices.size();

    vect.SetSize(size);

    for (int i = 0; i < size; ++i)
    {
        assert(indices[i] >= 0);
        assert(indices[i] < size_);

        vect[i] = (*this)[indices[i]];
    }
}

template <typename T>
void VectorView<T>::Print(const std::string& label, std::ostream& out) const
{
    out << label << "\n";

    for (int i = 0; i < size_; ++i)
    {
        out << data_[i] << "\n";
    }

    out << "\n";
}

template <typename T>
VectorView<T>& VectorView<T>::Add(double alpha, const VectorView<T>& rhs)
{
    assert(rhs.size_ == size_);

    for (int i = 0; i < size_; ++i)
    {
        data_[i] += alpha * rhs[i];
    }

    return *this;
}

template <typename T>
VectorView<T>& VectorView<T>::Add(const VectorView<T>& rhs)
{
    (*this) += rhs;

    return *this;
}

template <typename T>
VectorView<T>& VectorView<T>::Sub(double alpha, const VectorView<T>& rhs)
{
    assert(rhs.size_ == size_);

    for (int i = 0; i < size_; ++i)
    {
        data_[i] -= alpha * rhs[i];
    }

    return *this;
}

template <typename T>
VectorView<T>& VectorView<T>::Sub(const VectorView<T>& rhs)
{
    (*this) -= rhs;

    return *this;
}

template <typename T>
VectorView<T>& VectorView<T>::Set(double alpha, const VectorView<T>& rhs)
{
    assert(rhs.size_ == size_);

    for (int i = 0; i < size_; ++i)
    {
        data_[i] = alpha * rhs[i];
    }

    return *this;
}

template <typename T>
VectorView<T>& VectorView<T>::Set(const VectorView<T>& rhs)
{
    (*this) = rhs;

    return *this;
}

template <typename T>
T VectorView<T>::Sum() const
{
    T start = 0;
    return std::accumulate(std::begin(*this), std::end(*this), start);
}

/*! @brief Compute the inner product two vectors x^T y
    @param lhs left hand side vector x
    @param rhs right hand side vector y
    @retval the inner product
*/
template <typename T, typename U>
double InnerProduct(const VectorView<T>& lhs, const VectorView<U>& rhs)
{
    return lhs.Mult(rhs);
}

template <typename T>
double VectorView<T>::L2Norm() const
{
    return std::sqrt(InnerProduct(*this, *this));
}

template <typename T>
T VectorView<T>::Mult(const VectorView<T>& vect) const
{
    assert(vect.size() == size());

    T start = 0.0;

    return std::inner_product(std::begin(*this), std::end(*this), std::begin(vect), start);
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const VectorView<T>& vect)
{
    std::string label = "";
    vect.Print(label, out);

    return out;
}

// Templated Free Functions
/*! @brief Compute the L2 norm of the vector
    @param vect the vector to compute the L2 norm of
    @retval the L2 norm
*/
template <typename T>
double L2Norm(const VectorView<T>& vect)
{
    return vect.L2Norm();
}

/*! @brief Compute the inner product two vectors x^T y
    @param lhs left hand side vector x
    @param rhs right hand side vector y
    @retval the inner product
*/
template <typename T, typename U>
double operator*(const VectorView<T>& lhs, const VectorView<U>& rhs)
{
    return InnerProduct(lhs, rhs);
}

/*! @brief Entrywise multiplication x_i = x_i * y_i
    @param lhs left hand side vector x
    @param rhs right hand side vector y
*/
template <typename T>
VectorView<T> operator*=(VectorView<T> lhs, const VectorView<T>& rhs)
{
    assert(lhs.size() == rhs.size());

    const int size = lhs.size();

    for (int i = 0; i < size; ++i)
    {
        lhs[i] *= rhs[i];
    }

    return lhs;
}

/*! @brief Entrywise division x_i = x_i / y_i
    @param lhs left hand side vector x
    @param rhs right hand side vector y
*/
template <typename T>
VectorView<T> operator/=(VectorView<T> lhs, const VectorView<T>& rhs)
{
    assert(lhs.size() == rhs.size());

    const int size = lhs.size();

    for (int i = 0; i < size; ++i)
    {
        assert(rhs[i] != 0.0);

        lhs[i] /= rhs[i];
    }

    return lhs;
}

/*! @brief Entrywise addition x_i = x_i - y_i
    @param lhs left hand side vector x
    @param rhs right hand side vector y
*/
template <typename T, typename U>
VectorView<T> operator+=(VectorView<T> lhs, const VectorView<U>& rhs)
{
    assert(lhs.size() == rhs.size());

    const int size = lhs.size();

    for (int i = 0; i < size; ++i)
    {
        lhs[i] += rhs[i];
    }

    return lhs;
}

/*! @brief Entrywise subtraction x_i = x_i - y_i
    @param lhs left hand side vector x
    @param rhs right hand side vector y
*/
template <typename T>
VectorView<T> operator-=(VectorView<T> lhs, const VectorView<T>& rhs)
{
    assert(lhs.size() == rhs.size());

    const int size = lhs.size();

    for (int i = 0; i < size; ++i)
    {
        lhs[i] -= rhs[i];
    }

    return lhs;
}

/*! @brief Multiply a vector by a scalar
    @param vect vector to multiply
    @param val value to scale by
*/
template <typename T>
VectorView<T> operator*=(VectorView<T> vect, T val)
{
    for (T& i : vect)
    {
        i *= val;
    }

    return vect;
}

/*! @brief Divide a vector by a scalar
    @param vect vector to multiply
    @param val value to scale by
*/
template <typename T>
VectorView<T> operator/=(VectorView<T> vect, T val)
{
    assert(val != 0);

    for (T& i : vect)
    {
        i /= val;
    }

    return vect;
}

/*! @brief Add a scalar to each entry
    @param lhs vector to add to
    @param val the value to add
*/
template <typename T>
VectorView<T> operator+=(VectorView<T> lhs, T val)
{
    for (T& i : lhs)
    {
        i += val;
    }

    return lhs;
}

/*! @brief Subtract a scalar from each entry
    @param lhs vector to add to
    @param val the value to subtract
*/
template <typename T>
VectorView<T> operator-=(VectorView<T> lhs, T val)
{
    for (T& i : lhs)
    {
        i -= val;
    }

    return lhs;
}

/*! @brief Check if two vectors are equal
    @param lhs left hand side vector
    @param rhs right hand side vector
    @retval true if vectors are close enough to equal
*/
template <typename T, typename U>
bool operator==(const VectorView<T>& lhs, const VectorView<U>& rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }

    const int size = lhs.size();
    constexpr double tol = 1e-12;

    for (int i = 0; i < size; ++i)
    {
        if (std::fabs(lhs[i] - rhs[i]) > tol)
        {
            return false;
        }
    }

    return true;
}

/*! @brief Compute the absolute value maximum entry in a vector
    @param vect vector to find the max
    @retval the maximum entry value
*/
template <typename T>
T AbsMax(const VectorView<T>& vect)
{
    assert(vect.size() > 0);

    const auto compare = [](const T& lhs, const T& rhs)
    {
        return std::fabs(lhs) < std::fabs(rhs);
    };

    return std::fabs(*std::max_element(std::begin(vect), std::end(vect), compare));
}

/*! @brief Compute the maximum entry value in a vector
    @param vect vector to find the max
    @retval the maximum entry value
*/
template <typename T>
T Max(const VectorView<T>& vect)
{
    assert(vect.size() > 0);

    return *std::max_element(std::begin(vect), std::end(vect));
}

/*! @brief Compute the minimum entry value in a vector
    @param vect vector to find the minimum
    @retval the minimum entry value
*/
template <typename T>
T Min(const VectorView<T>& vect)
{
    assert(vect.size() > 0);

    return *std::min_element(std::begin(vect), std::end(vect));
}

/*! @brief Compute the absolute value minimum entry in a vector
    @param vect vector to find the minimum
    @retval the minimum entry value
*/
template <typename T>
T AbsMin(const VectorView<T>& vect)
{
    assert(vect.size() > 0);

    const auto compare = [](const T& lhs, const T& rhs)
    {
        return std::fabs(lhs) < std::fabs(rhs);
    };

    return std::fabs(*std::min_element(std::begin(vect), std::end(vect), compare));
}

/*! @brief Compute the sum of all vector entries
    @param vect vector to find the sum
    @retval the sum of all entries
*/
template <typename T>
T Sum(const VectorView<T>& vect)
{
    return vect.Sum();
}

/*! @brief Compute the mean of all vector entries
    @param vect vector to find the mean
    @retval the mean of all entries
*/
template <typename T>
double Mean(const VectorView<T>& vect)
{
    return Sum(vect) / (double) vect.size();
}

/*! @brief Print an std vector to a stream
    @param out stream to print to
    @param vect the vector to print
*/
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vect)
{
    out << "\n";

    for (const auto& i : vect)
    {
        out << i << "\n";
    }

    out << "\n";

    return out;
}

/*! @brief Randomize the entries in a double vector
    @param vect vector to randomize
    @param lo lower range limit
    @param hi upper range limit
    @param seed seed to rng, if positive
*/
void Randomize(VectorView<double> vect, double lo = 0.0, double hi = 1.0, int seed = -1);

/*! @brief Randomize the entries in a integer vector
    @param vect vector to randomize
    @param lo lower range limit
    @param hi upper range limit
    @param seed seed to rng, if positive
*/
void Randomize(VectorView<int> vect, int lo = 0, int hi = 1, int seed = -1);

template <typename T>
void VectorView<T>::Randomize(int lo, int hi, int seed)
{
    linalgcpp::Randomize(*this, lo, hi, seed);
}

/*! @brief Normalize a vector such that its L2 norm is 1.0
    @param vect vector to normalize
*/
void Normalize(VectorView<double> vect);

template <typename T>
void VectorView<T>::Normalize()
{
    throw std::runtime_error("Cannot normalize unless double!");
}

/*! @brief Subtract a constant vector set to the average
    from this vector: x_i = x_i - mean(x)
    @param vect vector to subtract average from
*/
void SubAvg(VectorView<double> vect);

/*! @brief Compute z = (alpha * x) + (beta * y) + (gamma * z)
    @param alpha x coefficient
    @param x x vector
    @param beta y coefficient
    @param y y vector
    @param gamma z coefficient
    @param z z vector
*/
template <typename T>
void Add(T alpha, const VectorView<T>& x, T beta, const VectorView<T>& y, T gamma, VectorView<T> z)
{
    int size = x.size();

    assert(y.size() == size);
    assert(z.size() == size);

    for (int i = 0; i < size; ++i)
    {
        z[i] = (alpha * x[i]) + (beta * y[i]) + (gamma * z[i]);
    }
}

/*! @brief Compute z = x + y
    @param x x vector
    @param y y vector
    @param z z vector
*/
template <typename T>
void Add(const VectorView<T>& x, const VectorView<T>& y, VectorView<T> z)
{
    Add(1.0, x, 1.0, y, 0.0, z);
}

/*! @brief Compute z = x - y
    @param x x vector
    @param y y vector
    @param z z vector
*/
template <typename T>
void Sub(const VectorView<T>& x, const VectorView<T>& y, VectorView<T> z)
{
    Add(1.0, x, -1.0, y, 0.0, z);
}

} // namespace linalgcpp

#endif // VECTORVIEW_HPP
