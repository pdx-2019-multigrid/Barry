/*! @file */

#ifndef SOLVERS_HPP__
#define SOLVERS_HPP__

#include "operator.hpp"
#include "vector.hpp"
#include "utilities.hpp"

namespace linalgcpp
{

/*! @class Tracks basic iterative solver information */
class Solver : public Operator
{
    public:
        /*! @brief Default Constructor */
        Solver();

        /*! @brief Copy Constructor */
        Solver(const Solver& other) noexcept = default;

        /*! @brief Move Constructor */
        Solver(Solver&& other) noexcept = default;

        /*! @brief Destructor */
        virtual ~Solver() noexcept = default;

        /*! @brief Swap tow solver */
        friend void swap(Solver& lhs, Solver& rhs) noexcept;

        /*! @brief Get number of iterations for last solve
            @retval number of iterations for last solve
        */
        virtual int GetNumIterations() const;

        /*! @brief Set verbose output
            @param verbose verbose output
        */
        virtual void SetVerbose(bool verbose);

        /*! @brief Set maximum iterations
            @param max_iter maximum iterations
        */
        virtual void SetMaxIter(int max_iter);

        /*! @brief Set relative tolerance
            @param reltol relative tolerance
        */
        virtual void SetRelTol(double rel_tol);

        /*! @brief Set absolute tolerance
            @param reltol absolute tolerance
        */
        virtual void SetAbsTol(double abs_tol);

        /*! @brief Get Operator A being applied
            @retval Operator A
        */
        virtual const Operator* GetOperator() const { return A_; };

        /*! @brief Set Operator A being applied, of same size
            @param Operator A
        */
        virtual void SetOperator(const Operator& A);

    protected:
        /*! @brief Constructor
            @param A operator to apply the action of A
            @param max_iter maxiumum number of iterations to perform
            @param rel_tol relative tolerance for stopping criteria
            @param abs_tol absolute tolerance for stopping criteria
            @param verbose display additional iteration information
            @param Dot Dot product function
        */
        Solver(const Operator& A, int max_iter, double rel_tol,
               double abs_tol, bool verbose,
               double (*Dot)(const VectorView<double>&, const VectorView<double>&) = linalgcpp::InnerProduct);
        const Operator* A_;

        int max_iter_;
        bool verbose_;
        double rel_tol_;
        double abs_tol_;

        double (*Dot_)(const VectorView<double>&, const VectorView<double>&);

        mutable int num_iter_;
};


class PCGSolver : public Solver
{
    public:
        /*! @brief Default Constructor */
        PCGSolver() = default;

        /*! @brief Unpreconditioned Constructor
            @param A operator to apply the action of A
            @param max_iter maxiumum number of iterations to perform
            @param rel_tol relative tolerance for stopping criteria
            @param abs_tol absolute tolerance for stopping criteria
            @param verbose display additional iteration information
            @param Dot Dot product function
        */
        PCGSolver(const Operator& A, int max_iter = 1000,
                 double rel_tol = 1e-16, double abs_tol = 1e-16, bool verbose = false,
                 double (*Dot)(const VectorView<double>&, const VectorView<double>&) = linalgcpp::InnerProduct);

        /*! @brief Preconditioned Constructor
            @param A operator to apply the action of A
            @param M operator to apply the preconditioner
            @param max_iter maxiumum number of iterations to perform
            @param rel_tol relative tolerance for stopping criteria
            @param abs_tol absolute tolerance for stopping criteria
            @param verbose display additional iteration information
            @param Dot Dot product function
        */
        PCGSolver(const Operator& A, const Operator& M, int max_iter = 1000,
                 double rel_tol = 1e-16, double abs_tol = 1e-16, bool verbose = false,
                 double (*Dot)(const VectorView<double>&, const VectorView<double>&) = linalgcpp::InnerProduct);

        /*! @brief Copy Constructor */
        PCGSolver(const PCGSolver& other) noexcept;

        /*! @brief Move Constructor */
        PCGSolver(PCGSolver&& other) noexcept;

        /*! @brief Assignment operator */
        PCGSolver& operator=(PCGSolver other) noexcept;

        /*! @brief Swap two solvers */
        friend void swap(PCGSolver& lhs, PCGSolver& rhs) noexcept;

        /*! @brief Default Destructor */
        virtual ~PCGSolver() noexcept = default;

        /*! @brief Set Preconditioner */
        void SetPreconditioner (const Operator& M) { M_ = &M; }

        /*! @brief Solve
            @param[in] input right hand side to solve for
            @param[in,out] output intial guess on input and solution on output
        */
        void Mult(const VectorView<double>& input, VectorView<double> output) const override;

    private:
        const Operator* M_;

        mutable Vector<double> Ap_;
        mutable Vector<double> r_;
        mutable Vector<double> p_;
        mutable Vector<double> z_;
};

class MINRESSolver : public Solver
{
    public:
        /*! @brief Default Constructor */
        MINRESSolver() = default;

        /*! @brief Constructor
            @param A operator to apply the action of A
            @param max_iter maxiumum number of iterations to perform
            @param rel_tol relative tolerance for stopping criteria
            @param abs_tol absolute tolerance for stopping criteria
            @param verbose display additional iteration information
            @param Dot Dot product function
        */
        MINRESSolver(const Operator& A, int max_iter = 1000,
                     double rel_tol = 1e-16, double abs_tol = 1e-16, bool verbose = false,
                     double (*Dot)(const VectorView<double>&, const VectorView<double>&) = linalgcpp::InnerProduct);

        /*! @brief Copy Constructor */
        MINRESSolver(const MINRESSolver& other) noexcept;

        /*! @brief Move Constructor */
        MINRESSolver(MINRESSolver&& other) noexcept;

        /*! @brief Assignment operator */
        MINRESSolver& operator=(MINRESSolver other) noexcept;

        /*! @brief Swap two solvers */
        friend void swap(MINRESSolver& lhs, MINRESSolver& rhs) noexcept;

        /*! @brief Default Destructor */
        virtual ~MINRESSolver() noexcept = default;

        /*! @brief Solve
            @param[in] input right hand side to solve for
            @param[in,out] output intial guess on input and solution on output
        */
        void Mult(const VectorView<double>& input, VectorView<double> output) const override;

    private:
        mutable Vector<double> w0_;
        mutable Vector<double> w1_;
        mutable Vector<double> v0_;
        mutable Vector<double> v1_;
        mutable Vector<double> q_;
};

class PMINRESSolver : public Solver
{
    public:
        /*! @brief Default Constructor */
        PMINRESSolver() = default;

        /*! @brief Constructor
            @param A operator to apply the action of A
            @param M operator to apply the preconditioner
            @param max_iter maxiumum number of iterations to perform
            @param rel_tol relative tolerance for stopping criteria
            @param abs_tol absolute tolerance for stopping criteria
            @param verbose display additional iteration information
            @param Dot Dot product function
        */
        PMINRESSolver(const Operator& A, const Operator& M, int max_iter = 1000,
                 double rel_tol = 1e-16, double abs_tol = 1e-16, bool verbose = false,
                 double (*Dot)(const VectorView<double>&, const VectorView<double>&) = linalgcpp::InnerProduct);

        /*! @brief Copy Constructor */
        PMINRESSolver(const PMINRESSolver& other) noexcept;

        /*! @brief Move Constructor */
        PMINRESSolver(PMINRESSolver&& other) noexcept;

        /*! @brief Assignment operator */
        PMINRESSolver& operator=(PMINRESSolver other) noexcept;

        /*! @brief Swap two solvers */
        friend void swap(PMINRESSolver& lhs, PMINRESSolver& rhs) noexcept;

        /*! @brief Default Destructor */
        virtual ~PMINRESSolver() noexcept = default;


        /*! @brief Solve
            @param[in] input right hand side to solve for
            @param[in,out] output intial guess on input and solution on output
        */
        void Mult(const VectorView<double>& input, VectorView<double> output) const override;

    private:
        const Operator* M_;

        mutable Vector<double> w0_;
        mutable Vector<double> w1_;
        mutable Vector<double> v0_;
        mutable Vector<double> v1_;
        mutable Vector<double> u1_;
        mutable Vector<double> q_;
};


/*! @brief Conjugate Gradient.  Solves Ax = b
    @param A operator to apply the action of A
    @param b right hand side vector
    @param num_iter number of iterations completed
    @param max_iter maxiumum number of iterations to perform
    @param rel_tol relative tolerance for stopping criteria
    @param abs_tol absolute tolerance for stopping criteria
    @param verbose display additional iteration information
    @retval Vector x the computed solution

    @note Uses random initial guess for x
*/
Vector<double> CG(const Operator& A, const VectorView<double>& b, int& num_iter, 
                  int max_iter = 1000, double rel_tol = 1e-16, double abs_tol = 1e-16,
                  bool verbose = false);

/*! @brief Conjugate Gradient.  Solves Ax = b for positive definite A
    @param A operator to apply the action of A
    @param b right hand side vector
    @param[in,out] x intial guess on input and solution on output
    @param max_iter maxiumum number of iterations to perform
    @param rel_tol relative tolerance for stopping criteria
    @param abs_tol absolute tolerance for stopping criteria
    @param verbose display additional iteration information
    @retval num_iter number of iterations completed
*/
int CG(const Operator& A, const VectorView<double>& b, VectorView<double> x,
        int max_iter = 1000, double rel_tol = 1e-16, double abs_tol = 1e-16,
        bool verbose = false);

/*! @brief Preconditioned Conjugate Gradient.  Solves Ax = b
           where M is preconditioner for A
    @param A operator to apply the action of A
    @param M operator to apply the preconditioner
    @param b right hand side vector
    @param num_iter number of iterations completed
    @param max_iter maxiumum number of iterations to perform
    @param rel_tol relative tolerance for stopping criteria
    @param abs_tol absolute tolerance for stopping criteria
    @param verbose display additional iteration information
    @retval Vector x the computed solution

    @note Uses random initial guess for x
*/
Vector<double> PCG(const Operator& A, const Operator& M, const VectorView<double>& b,
                   int& num_iter, int max_iter = 1000, double rel_tol = 1e-16, 
                   double abs_tol = 1e-16,bool verbose = false);

/*! @brief Preconditioned Conjugate Gradient.  Solves Ax = b
           where M is preconditioner for A
    @param A operator to apply the action of A
    @param M operator to apply the preconditioner
    @param b right hand side vector
    @param[in,out] x intial guess on input and solution on output
    @param max_iter maxiumum number of iterations to perform
    @param rel_tol relative tolerance for stopping criteria
    @param abs_tol absolute tolerance for stopping criteria
    @param verbose display additional iteration information
    @retval num_iter number of iterations completed
*/
int PCG(const Operator& A, const Operator& M, const VectorView<double>& b, VectorView<double> x,
         int max_iter = 1000, double rel_tol = 1e-16, double abs_tol = 1e-16,
         bool verbose = false);

/*! @brief MINRES.  Solves Ax = b for symmetric A
    @param A operator to apply the action of A
    @param b right hand side vector
    @param max_iter maxiumum number of iterations to perform
    @param rel_tol relative tolerance for stopping criteria
    @param abs_tol absolute tolerance for stopping criteria
    @param verbose display additional iteration information
    @retval Vector x the computed solution

    Modified from mfem implementation
    @note Uses random initial guess for x
*/
Vector<double> MINRES(const Operator& A, const VectorView<double>& b,
                      int max_iter = 1000, double rel_tol = 1e-16, double abs_tol = 1e-16,
                      bool verbose = false);

/*! @brief MINRES.  Solves Ax = b for symmetric A
    @param A operator to apply the action of A
    @param b right hand side vector
    @param[in,out] x intial guess on input and solution on output
    @param max_iter maxiumum number of iterations to perform
    @param rel_tol relative tolerance for stopping criteria
    @param abs_tol absolute tolerance for stopping criteria
    @param verbose display additional iteration information

    Modified from mfem implementation
*/

void MINRES(const Operator& A, const VectorView<double>& b, VectorView<double> x,
            int max_iter = 1000, double rel_tol = 1e-16, double abs_tol = 1e-16,
            bool verbose = false);

/*! @brief Preconditioned MINRES.  Solves Ax = b for symmetric A
    @param A operator to apply the action of A
    @param M operator to apply of the preconditioner
    @param b right hand side vector
    @param max_iter maxiumum number of iterations to perform
    @param rel_tol relative tolerance for stopping criteria
    @param abs_tol absolute tolerance for stopping criteria
    @param verbose display additional iteration information
    @retval Vector x the computed solution

    Modified from mfem implementation
    @note Uses random initial guess for x
*/
Vector<double> PMINRES(const Operator& A, const Operator& M, const VectorView<double>& b,
                       int max_iter = 1000, double rel_tol = 1e-16, double abs_tol = 1e-16,
                       bool verbose = false);

/*! @brief Preconditioned MINRES.  Solves Ax = b for symmetric A
    @param A operator to apply the action of A
    @param M operator to apply of the preconditioner
    @param b right hand side vector
    @param[in,out] x intial guess on input and solution on output
    @param max_iter maxiumum number of iterations to perform
    @param rel_tol relative tolerance for stopping criteria
    @param abs_tol absolute tolerance for stopping criteria
    @param verbose display additional iteration information

    Modified from mfem implementation
*/
void PMINRES(const Operator& A, const Operator& M, const VectorView<double>& b, VectorView<double> x,
             int max_iter = 1000, double rel_tol = 1e-16, double abs_tol = 1e-16,
             bool verbose = false);

} //namespace linalgcpp

#endif // SOLVERS_HPP__
