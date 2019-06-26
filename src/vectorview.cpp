#include "vectorview.hpp"

namespace linalgcpp
{
void Randomize(VectorView<double> vect, double lo, double hi, int seed)
{
    std::random_device r;
    std::default_random_engine dev(r());
    std::uniform_real_distribution<double> gen(lo, hi);

    if (seed >= 0)
    {
        dev.seed(seed);
    }

    for (double& val : vect)
    {
        val = gen(dev);
    }
}

void Randomize(VectorView<int> vect, int lo, int hi, int seed)
{
    std::random_device r;
    std::default_random_engine dev(r());
    std::uniform_int_distribution<int> gen(lo, hi);

    if (seed >= 0)
    {
        dev.seed(seed);
    }

    for (int& val : vect)
    {
        val = gen(dev);
    }
}

void Normalize(VectorView<double> vect)
{
    vect /= L2Norm(vect);
}

void SubAvg(VectorView<double> vect)
{
    vect -= Mean(vect);
}

template <>
void VectorView<double>::Normalize()
{
    linalgcpp::Normalize(*this);
}

} // namespace linalgcpp
