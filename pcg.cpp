#include "vectorview.hpp"
#include "parser.hpp"

using namespace linalgcpp;

// Regular conjugate gradient method.
// Returns number of iterations.
int CG(const SparseMatrix<double>& A,
       Vector<double>& x,
       const Vector<double>& x0,
       const Vector<double>& b,
       double tol = 1e-9,
       bool verbose = true);

// Preconditioned conjugate gradient method.
// Returns number of iterations.
int PCG(const SparseMatrix<double>& A,
        Vector<double>& x,
        const Vector<double>& x0,
        const Vector<double>& b,
        Vector<double>(SparseMatrix<double>::*precond)
        (Vector<double>) const,
        double tol = 1e-9,
        bool verbose = true);

// Conjugate gradient method with two-level preconditioner.
// Returns number of iterations.
int TL(const SparseMatrix<double>& A,
           Vector<double>& x,
           const Vector<double>& x0,
           const Vector<double>& b,
           Vector<double>(*precond)
           (const SparseMatrix<double>&,
            const Vector<double>&,
            const SparseMatrix<int>&,
            const SparseMatrix<double>&),
           double tol = 1e-9,
           bool verbose = true);

// Two-level preconditioner used in function TL.
Vector<double> TwoLevel(const SparseMatrix<double>& A, 
                        const Vector<double>& b,
                        const SparseMatrix<int>& P,
                        const SparseMatrix<double>& Ac);

int main()
{
  // Create graph Laplacian from edge list.
  // Must have zeroth vertex.
  std::string filename;
  std::cout << "Please type the input filename: ";
  getline(std::cin, filename);
  SparseMatrix<double> A(ReadGraphList(filename));

  // Make the matrix positive definite.
  A.EliminateRowCol(A.Rows() - 1);
  A.EliminateZeros();

  // Let n be length of a vector.
  const int n = A.Cols();

  // Compare size of matrix with number of iterations
  std::cout << "The matrix is " << n << "x" << n 
            << ".\n" <<std::endl;

  Vector<double> x0(n); // Exact solution x0.
  Randomize(x0, -10.0, 10.0); // Fill x0 with random values.
  Vector<double> b(A.Mult(x0)); // Definition of b.

  Vector<double> x(n); // Iterate x.
  int num_iter;

  std::cout << "CG: ";
  CG(A, x, x0, b);

  // Each preconditioner is a method of the SparseMatrix
  // class. The syntax leaves something to be desired.
  std::cout << "PCG Jacobi: ";
  PCG(A, x, x0, b, &SparseMatrix<double>::Jacobi);

  std::cout << "PCG l1-smoother: ";
  PCG(A, x, x0, b, &SparseMatrix<double>::L1);

  std::cout << "PCG Gauss-Seidel: ";
  PCG(A, x, x0, b, &SparseMatrix<double>::GaussSeidel);

  return 0;
}
// This function solves Ax = b.
int CG(const SparseMatrix<double>& A,
       Vector<double>& x,
       const Vector<double>& x0,
       const Vector<double>& b,
       double tol,
       bool verbose)
{ 
  x = 0; // Set initial interate to zero.
  // Because x = 0, the first residual r = b - A(x) = b. 
  Vector<double> r(b);
  Vector<double> p(r); // Initial search direction.
  Vector<double> g; // See usage below.
  int num_iter = 0;
  double c0 = r.Mult(r); // r dot r
  Vector<double> c(A.Cols() + 2); // Squares of residual norm.
  c[0] = c0;

  double alpha, beta, c1, t;

  // Beginning of CG algorithm.
  for (int i = 0; i < A.Rows() + 1; ++i)
  {
    A.Mult(p, g); // g := Ap.
    t = p.Mult(g);
    alpha = c[i] / t;

    x.Add(alpha, p);
    r.Sub(alpha, g);

    c1 = c[i];
    c[i + 1]  = r.Mult(r);

    ++num_iter;
    if (c[i + 1] < tol * tol * c0)
      break;

    beta = c[i + 1] / c1;
    p *= beta;
    p += r;
  }

  if (verbose)
  {
    // If num_iter > A.Cols(), then the algorithm
    // did not converge in the expected (theoretical)
    // number of iterations.
    printf("num_iter = %d\n", num_iter);   
 
    // Print the last three residual norms 
    // that are computed.
    for (int i = num_iter - 2; i < num_iter + 1; ++i)
      printf("|r| = %.3e\n", sqrt(c[i]));

    // Let us see how close the approximation is
    // in the euclidean norm.
    r = x - x0;
    double error(L2Norm(r));

    std::cout << "Compare the approx soln with the exact: ";
    printf("|x - x0| = %.3e\n", error);
    std::cout << std::endl;
  }
  return num_iter;
}

// This function solves Ax = b using one of the three
// preconditioners shown in the main.
int PCG(const SparseMatrix<double>& A,
          Vector<double>& x,
          const Vector<double>& x0,
          const Vector<double>& b,
          Vector<double>(SparseMatrix<double>::*precond)
          (Vector<double>) const,
          double tol,
          bool verbose)
{ 
  x = 0; // Set initial interate to zero.
  // Because x = 0, the first residual r = b - A(x) = b. 
  Vector<double> r(b);
  Vector<double> y = (A.*precond)(r); // Preconditioned residual.
  Vector<double> p(y); // Initial search direction.
  Vector<double> g; // See usage below.
  int num_iter = 0;
  double c0 = r.Mult(y); // r dot y
  Vector<double> c(A.Cols() + 2); // Squares of residual norm.
  c[0] = c0;

  double alpha, beta, c1, t;

  // Beginning of PCG algorithm.
  for (int i = 0; i < A.Rows() + 1; ++i)
  {
    A.Mult(p, g); // g := Ap.
    t = p.Mult(g);
    alpha = c[i] / t;

    x.Add(alpha, p);
    r.Sub(alpha, g);

    // Copy two vectors by value. May be inefficient.
    y = (A.*precond)(r);

    c1 = c[i];
    c[i + 1]  = r.Mult(y);

    ++num_iter;
    if (c[i + 1] < tol * tol * c0)
      break;

    beta = c[i + 1] / c1;
    p *= beta;
    p += y;
  }

  if (verbose)
  {
    // If num_iter > A.Cols(), then the algorithm
    // did not converge in the expected (theoretical)
    // number of iterations.
    printf("num_iter = %d\n", num_iter);   
 
    // Print the last three residual norms 
    // that are computed.
    for (int i = num_iter - 2; i < num_iter + 1; ++i)
      printf("|r| = %.3e\n", sqrt(c[i]));

    // Let us see how close the approximation is
    // in the euclidean norm.
    r = x - x0;
    double error(L2Norm(r));

    std::cout << "Compare the approx soln with the exact: ";
    printf("|x - x0| = %.3e\n", error);
    std::cout << std::endl;
  }
  return num_iter;
}

int TL(const SparseMatrix<double>& A,
           Vector<double>& x,
           const Vector<double>& x0,
           const Vector<double>& b,
           Vector<double>(*precond)
           (const SparseMatrix<double>&,
            const Vector<double>&,
            const SparseMatrix<int>&,
            const SparseMatrix<double>&),
           bool verbose,
           double tol)
{ 
  // Determine interpolation matrix P and coarse graph Laplacian Ac.
  int nparts = std::max(2.0, cbrt(A.Cols()));
  SparseMatrix<int> P(Unweighted(Partition(A, nparts)));
  SparseMatrix<double> Ac = P.Transpose().Mult(A.Mult(P));
  Ac.EliminateZeros();

  x = 0; // Set initial iterate to zero.
  // Because x = 0, the first residual r = b - A(x) = b. 
  Vector<double> r(b);
  Vector<double> y = (*precond)(A, r, P, Ac); // Preconditioned residual.
  Vector<double> p(y); // Initial search direction.
  Vector<double> g; // See usage below.
  int num_iter = 0;
  double c0 = r.Mult(y); // r dot y
  Vector<double> c(A.Cols() + 2); // Squares of residual norm.
  c[0] = c0;

  double alpha, beta, c1, t;

  // Beginning of PCG algorithm.
  for (int i = 0; i < A.Rows() + 1; ++i)
  {
    A.Mult(p, g); // g := Ap.
    t = p.Mult(g);
    alpha = c[i] / t;

    x.Add(alpha, p);
    r.Sub(alpha, g);

    // Copy two vectors by value. May be inefficient.
    y = (*precond)(A, r, P, Ac);

    c1 = c[i];
    c[i + 1]  = r.Mult(y);

    ++num_iter;
    if (c[i + 1] < tol * tol * c0)
      break;

    beta = c[i + 1] / c1;
    p *= beta;
    p += y;
  }

  if (verbose)
  {
    // If num_iter > A.Cols(), then the algorithm
    // did not converge in the expected (theoretical)
    // number of iterations.
    printf("num_iter = %d\n", num_iter);
    printf("nparts = %d\n", nparts);

    // Print the last three residual norms 
    // that are computed.
    for (int i = num_iter - 2; i < num_iter + 1; ++i)
      printf("|r| = %.3e\n", sqrt(c[i]));

    // Let us see how close the approximation is
    // in the euclidean norm.
    r = x - x0;
    double error(L2Norm(r));

    std::cout << "Compare the approx soln with the exact: ";
    printf("|x - x0| = %.3e\n", error);
    std::cout << std::endl;
  }
  return num_iter;
}

Vector<double> TwoLevel(const SparseMatrix<double>& A, 
                        const Vector<double>& b,
                        const SparseMatrix<int>& P,
                        const SparseMatrix<double>& Ac)
{
  Vector<double> x(A.ForwardGauss(b)); // Use M = D + L. 
  Vector<double> rc(P.MultAT(b - A.Mult(x))); // x is x_(1/3)
  Vector<double> xc(Ac.Cols());
  CG(Ac, xc, b, rc, false);
  x.Add(P.Mult(xc));
  x.Add(A.BackwardGauss(b - A.Mult(x))); // M.Transpose() = D + U // x is x_(2/3)

  return x;
}