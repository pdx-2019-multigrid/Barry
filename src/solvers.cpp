#include "solvers.hpp"

namespace linalgcpp
{

Solver::Solver()
    : A_(nullptr), max_iter_(0), verbose_(false),
      rel_tol_(0.0), abs_tol_(0.0), Dot_(nullptr),
      num_iter_(0)
{
}

Solver::Solver(const Operator& A, int max_iter, double rel_tol,
               double abs_tol, bool verbose,
               double (*Dot)(const VectorView<double>&, const VectorView<double>&))
    : Operator(A), A_(&A), max_iter_(max_iter), verbose_(verbose),
      rel_tol_(rel_tol), abs_tol_(abs_tol), Dot_(Dot),
      num_iter_(0)
{
    assert(A_);
    assert(Dot_);
    assert(max_iter_ >= 0);
    assert(A_->Rows() == A_->Cols());
}

void swap(Solver& lhs, Solver& rhs) noexcept
{
    swap(static_cast<Operator&>(lhs), static_cast<Operator&>(rhs));

    std::swap(lhs.A_, rhs.A_);
    std::swap(lhs.max_iter_, rhs.max_iter_);
    std::swap(lhs.num_iter_, rhs.num_iter_);
    std::swap(lhs.verbose_, rhs.verbose_);
    std::swap(lhs.rel_tol_, rhs.rel_tol_);
    std::swap(lhs.abs_tol_, rhs.abs_tol_);
    std::swap(lhs.Dot_, rhs.Dot_);
}

int Solver::GetNumIterations() const
{
    return num_iter_;
}

void Solver::SetVerbose(bool verbose)
{
    verbose_ = verbose;
}

void Solver::SetMaxIter(int max_iter)
{
    max_iter_ = max_iter;
}

void Solver::SetRelTol(double rel_tol)
{
    rel_tol_ = rel_tol;
}

void Solver::SetAbsTol(double abs_tol)
{
    abs_tol_ = abs_tol;
}

void Solver::SetOperator(const Operator& A)
{
    assert(A.Rows() == A_->Rows());
    assert(A.Cols() == A_->Cols());

    A_ = &A;
}

Vector<double> CG(const Operator& A, const VectorView<double>& b, int& num_iter,
                  int max_iter, double rel_tol, double abs_tol, bool verbose)
{
    Vector<double> x(A.Rows());
    Randomize(x);

    num_iter = CG(A, b, x, max_iter, rel_tol, abs_tol, verbose);

    return x;
}

int CG(const Operator& A, const VectorView<double>& b, VectorView<double> x,
        int max_iter, double rel_tol, double abs_tol, bool verbose)
{
    PCGSolver cg(A, max_iter, rel_tol, abs_tol, verbose);
    cg.Mult(b, x);
    return cg.GetNumIterations();
}

PCGSolver::PCGSolver(const Operator& A, int max_iter,
                     double rel_tol, double abs_tol, bool verbose,
                     double (*Dot)(const VectorView<double>&, const VectorView<double>&))
    : Solver(A, max_iter, rel_tol, abs_tol, verbose, Dot), M_(nullptr),
      Ap_(A_->Rows()), r_(A_->Rows()), p_(A_->Rows()), z_(A_->Rows())
{
    assert(A_);

    assert(A_->Rows() == A_->Cols());
}

PCGSolver::PCGSolver(const Operator& A, const Operator& M, int max_iter,
                     double rel_tol, double abs_tol, bool verbose,
                     double (*Dot)(const VectorView<double>&, const VectorView<double>&))
    : Solver(A, max_iter, rel_tol, abs_tol, verbose, Dot), M_(&M),
      Ap_(A_->Rows()), r_(A_->Rows()), p_(A_->Rows()), z_(A_->Rows())
{
    assert(A_);
    assert(M_);

    assert(A_->Rows() == A_->Cols());
    assert(A_->Rows() == M_->Cols());
    assert(M_->Rows() == M_->Cols());
}

PCGSolver::PCGSolver(const PCGSolver& other) noexcept
    : Solver(other), M_(other.M_), Ap_(other.Ap_), r_(other.r_), p_(other.p_),
      z_(other.z_)
{
}

PCGSolver::PCGSolver(PCGSolver&& other) noexcept
{
    swap(*this, other);
}

PCGSolver& PCGSolver::operator=(PCGSolver other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(PCGSolver& lhs, PCGSolver& rhs) noexcept
{
    swap(static_cast<Solver&>(lhs), static_cast<Solver&>(rhs));

    std::swap(lhs.M_, rhs.M_);
    std::swap(lhs.Ap_, rhs.Ap_);
    std::swap(lhs.r_, rhs.r_);
    std::swap(lhs.p_, rhs.p_);
    std::swap(lhs.z_, rhs.z_);
}

void PCGSolver::Mult(const VectorView<double>& b, VectorView<double> x) const
{
    assert(A_);

    assert(x.size() == A_->Rows());
    assert(b.size() == A_->Rows());

    A_->Mult(x, Ap_);
    r_ = b;
    r_ -= Ap_;

    if (M_)
    {
        M_->Mult(r_, z_);
    }
    else
    {
        z_ = r_;
    }

    p_ = z_;

    const double r0 = (*Dot_)(z_, r_);
    //const double r0 = (*Dot_)(r_, r_);

    const double tol_tol = std::max(r0 * rel_tol_ * rel_tol_, abs_tol_ * abs_tol_);

    num_iter_ = 1;

    if ((*Dot_)(z_, r_) < tol_tol)
    //if ((*Dot_)(r_, r_) < tol_tol)
    {
        return;
    }

    do
    {
        A_->Mult(p_, Ap_);

        double pAp = (*Dot_)(p_, Ap_);
        double r_z = (*Dot_)(r_ , z_);
        double alpha = r_z / pAp;

        linalgcpp_verify(pAp > -1e12, "PCG is not positive definite!");

        x.Add(alpha, p_);

        double denom = (*Dot_)(z_, r_);

        r_.Sub(alpha, Ap_);

        if (M_)
        {
            M_->Mult(r_, z_);
        }
        else
        {
            z_ = r_;
        }

        double numer = (*Dot_)(z_, r_);

        if (verbose_)
        {
            printf("PCG %d: %.2e\n", num_iter_, numer / r0);
        }

        if (numer < tol_tol)
        {
            break;
        }

        double beta = numer / denom;

        p_ *= beta;
        p_ += z_;
    }
    while (++num_iter_ < max_iter_);
}

Vector<double> PCG(const Operator& A, const Operator& M, const VectorView<double>& b,
                   int& num_iter, int max_iter, double rel_tol, double abs_tol, 
                   bool verbose)
{
    Vector<double> x(A.Rows());
    Randomize(x);

    num_iter = PCG(A, M, b, x, max_iter, rel_tol, abs_tol, verbose);
    return x;
}

int PCG(const Operator& A, const Operator& M, const VectorView<double>& b, VectorView<double> x,
         int max_iter, double rel_tol, double abs_tol, bool verbose)
{
    PCGSolver pcg(A, M, max_iter, rel_tol, abs_tol, verbose);
    pcg.Mult(b, x);
    return pcg.GetNumIterations();
}

MINRESSolver::MINRESSolver(const Operator& A, int max_iter, double rel_tol, double abs_tol, bool verbose,
                           double (*Dot)(const VectorView<double>&, const VectorView<double>&))
    : Solver(A, max_iter, rel_tol, abs_tol, verbose, Dot),
      w0_(A_->Rows()), w1_(A_->Rows()),
      v0_(A_->Rows()), v1_(A_->Rows()),
      q_(A_->Rows())
{
    assert(A_->Rows() == A_->Cols());
}

MINRESSolver::MINRESSolver(const MINRESSolver& other) noexcept
    : Solver(other), w0_(other.w0_), w1_(other.w1_), v0_(other.v0_),
      v1_(other.v1_), q_(other.q_)
{
}

MINRESSolver::MINRESSolver(MINRESSolver&& other) noexcept
{
    swap(*this, other);
}

MINRESSolver& MINRESSolver::operator=(MINRESSolver other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(MINRESSolver& lhs, MINRESSolver& rhs) noexcept
{
    swap(static_cast<Solver&>(lhs), static_cast<Solver&>(rhs));

    std::swap(lhs.w0_, rhs.w0_);
    std::swap(lhs.w1_, rhs.w1_);
    std::swap(lhs.v0_, rhs.v0_);
    std::swap(lhs.v1_, rhs.v1_);
    std::swap(lhs.q_, rhs.q_);
}

void MINRESSolver::Mult(const VectorView<double>& b, VectorView<double> x) const
{
    assert(A_);
    assert(x.size() == A_->Rows());
    assert(b.size() == A_->Rows());

    const int size = A_->Cols();

    w0_ = 0.0;
    w1_ = 0.0;
    v0_ = 0.0;

    A_->Mult(x, q_);
    v1_ = b;
    v1_ -= q_;

    double beta = std::sqrt((*Dot_)(v1_, v1_));
    double eta = beta;

    double gamma = 1.0;
    double gamma2 = 1.0;

    double sigma = 0;
    double sigma2 = 0;

    double tol = std::max(beta * rel_tol_ * rel_tol_, abs_tol_ * abs_tol_);

    for (num_iter_ = 1; num_iter_ <= max_iter_; ++num_iter_)
    {
        v1_ /= beta;
        A_->Mult(v1_, q_);

        const double alpha = (*Dot_)(v1_, q_);

        for (int i = 0; i < size; ++i)
        {
            v0_[i] = q_[i] - (beta * v0_[i]) - (alpha * v1_[i]);
        }

        const double delta = gamma2 * alpha - gamma * sigma2 * beta;
        const double rho3 = sigma * beta;
        const double rho2 = sigma2 * alpha + gamma * gamma2 * beta;

        beta = std::sqrt((*Dot_)(v0_, v0_));

        const double rho1 = std::sqrt((delta * delta) + (beta * beta));

        for (int i = 0; i < size; ++i)
        {
            w0_[i] = ((1.0 / rho1) * v1_[i]) - ( (rho3 / rho1)  * w0_[i]) - (( rho2 / rho1) * w1_[i]);
        }

        gamma = gamma2;
        gamma2 = delta / rho1;

        for (int i = 0; i < size; ++i)
        {
            x[i] += gamma2 * eta * w0_[i];
        }

        sigma = sigma2;
        sigma2 = beta / rho1;

        eta = -sigma2 * eta;

        if (verbose_)
        {
            printf("MINRES %d: %.2e\n", num_iter_, eta);
        }

        if (std::fabs(eta) < tol)
        {
            break;
        }

        swap(v0_, v1_);
        swap(w0_, w1_);
    }
}

Vector<double> MINRES(const Operator& A, const VectorView<double>& b,
                      int max_iter, double rel_tol, double abs_tol, bool verbose)
{
    Vector<double> x(A.Rows());
    Randomize(x);

    MINRES(A, b, x, max_iter, rel_tol, abs_tol, verbose);

    return x;
}

void MINRES(const Operator& A, const VectorView<double>& b, VectorView<double> x,
            int max_iter, double rel_tol, double abs_tol, bool verbose)
{
    MINRESSolver minres(A, max_iter, rel_tol, abs_tol, verbose);

    minres.Mult(b, x);
}

PMINRESSolver::PMINRESSolver(const Operator& A, const Operator& M, int max_iter,
                             double rel_tol, double abs_tol, bool verbose,
                             double (*Dot)(const VectorView<double>&, const VectorView<double>&))
    : Solver(A, max_iter, rel_tol, abs_tol, verbose, Dot), M_(&M),
      w0_(A_->Rows()), w1_(A_->Rows()),
      v0_(A_->Rows()), v1_(A_->Rows()),
      u1_(A_->Rows()), q_(A_->Rows())
{
    assert(A_);
    assert(M_);

    assert(A_->Rows() == A_->Cols());
    assert(A_->Rows() == M_->Cols());
    assert(M_->Rows() == M_->Cols());
}

PMINRESSolver::PMINRESSolver(const PMINRESSolver& other) noexcept
    : Solver(other), M_(other.M_), w0_(other.w0_), w1_(other.w1_),
      v0_(other.v0_), v1_(other.v1_), u1_(other.u1_), q_(other.q_)
{
}

PMINRESSolver::PMINRESSolver(PMINRESSolver&& other) noexcept
{
    swap(*this, other);
}

PMINRESSolver& PMINRESSolver::operator=(PMINRESSolver other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(PMINRESSolver& lhs, PMINRESSolver& rhs) noexcept
{
    swap(static_cast<Solver&>(lhs), static_cast<Solver&>(rhs));

    std::swap(lhs.M_, rhs.M_);
    std::swap(lhs.w0_, rhs.w0_);
    std::swap(lhs.w1_, rhs.w1_);
    std::swap(lhs.v0_, rhs.v0_);
    std::swap(lhs.v1_, rhs.v1_);
    std::swap(lhs.u1_, rhs.u1_);
    std::swap(lhs.q_, rhs.q_);
}

void PMINRESSolver::Mult(const VectorView<double>& b, VectorView<double> x) const
{
    assert(A_);
    assert(M_);

    assert(b.size() == A_->Rows());
    assert(x.size() == A_->Cols());

    const int size = A_->Cols();

    w0_ = 0.0;
    w1_ = 0.0;
    v0_ = 0.0;

    A_->Mult(x, q_);
    v1_ = b;
    v1_ -= q_;

    M_->Mult(v1_, u1_);

    double beta = std::sqrt((*Dot_)(u1_, v1_));
    double eta = beta;

    double gamma = 1.0;
    double gamma2 = 1.0;

    double sigma = 0;
    double sigma2 = 0;

    double tol = std::max(beta * rel_tol_ * rel_tol_, abs_tol_ * abs_tol_);

    for (num_iter_ = 1; num_iter_ <= max_iter_; ++num_iter_)
    {
        v1_ /= beta;
        u1_ /= beta;

        A_->Mult(u1_, q_);

        const double alpha = (*Dot_)(u1_, q_);

        for (int i = 0; i < size; ++i)
        {
            v0_[i] = q_[i] - (beta * v0_[i]) - (alpha * v1_[i]);
        }

        const double delta = gamma2 * alpha - gamma * sigma2 * beta;
        const double rho3 = sigma * beta;
        const double rho2 = sigma2 * alpha + gamma * gamma2 * beta;

        M_->Mult(v0_, q_);
        beta = std::sqrt((*Dot_)(v0_, q_));

        const double rho1 = std::sqrt((delta * delta) + (beta * beta));

        for (int i = 0; i < size; ++i)
        {
            w0_[i] = ((1.0 / rho1) * u1_[i]) - ( (rho3 / rho1)  * w0_[i]) - (( rho2 / rho1) * w1_[i]);
        }

        gamma = gamma2;
        gamma2 = delta / rho1;

        for (int i = 0; i < size; ++i)
        {
            x[i] += gamma2 * eta * w0_[i];
        }

        sigma = sigma2;
        sigma2 = beta / rho1;

        eta = -sigma2 * eta;

        if (verbose_)
        {
            printf("PMINRES %d: %.2e\n", num_iter_, eta);
        }

        if (std::fabs(eta) < tol)
        {
            break;
        }

        swap(u1_, q_);
        swap(v0_, v1_);
        swap(w0_, w1_);
    }
}

Vector<double> PMINRES(const Operator& A, const Operator& M, const VectorView<double>& b,
                       int max_iter, double rel_tol, double abs_tol, bool verbose)
{
    Vector<double> x(A.Rows());
    Randomize(x);

    PMINRES(A, M, b, x, max_iter, rel_tol, abs_tol, verbose);

    return x;
}

void PMINRES(const Operator& A, const Operator& M, const VectorView<double>& b, VectorView<double> x,
             int max_iter, double rel_tol, double abs_tol, bool verbose)
{
    PMINRESSolver pminres(A, M, max_iter, rel_tol, abs_tol, verbose);

    pminres.Mult(b, x);
}

} //namespace linalgcpp
