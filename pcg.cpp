#include "partition.hpp"
using namespace linalgcpp;

// This file takes an edge list for a connected graph and creates a graph Laplacian. 
// Then a row and column is deleted to make the matrix positive definite.
// A random vector x0 is used to set a right-hand side vector b.
// We solve Ax = b using preconditoned conjugate gradient.
// The output includes the last three residual norms, the number of iterations
// and the distance between the exact solution and approximate one.

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

// Conjugate gradient method with multilevel preconditioner.
// Returns number of iterations.
int ML(const SparseMatrix<double>& A0,
       Vector<double>& x,
       const Vector<double>& x0,
       const Vector<double>& b,
       Vector<double>(*precond)
       (const std::vector<SparseMatrix<int>>&,
        const std::vector<SparseMatrix<double>>&,
        const Vector<double>&,
        const int,
        int k),
        int ncoarse,
        int max_level,
        bool verbose = true,
        double tol = 1e-9);

// Multilevel preconditioner used in function ML.
Vector<double> Multilevel(const std::vector<SparseMatrix<int>>& P,
                          const std::vector<SparseMatrix<double>>& A,
                          const Vector<double>& b,
                          const int L, // Number of levels
                          int k); // Current level index

// Get sequence of interpolation matrices P_k
// and course graph laplacians A_k.
int GetSequence(std::vector<SparseMatrix<int>>& P,
                std::vector<SparseMatrix<double>>& A,
                int ncoarse,
                double q,
                int k = 0);


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

  std::cout << "TL: ";
  TL(A, x, x0, b, TwoLevel);

  std::cout << "ML: ";
  ML(A, x, x0, b, Multilevel, cbrt(n), 10);

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
       double tol,
       bool verbose)
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
    printf("nparts = %d, num_iter = %d\n", nparts, num_iter);

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
  Vector<double> x(A.GaussSeidel(b)); 
  Vector<double> rc(P.MultAT(b - A.Mult(x))); // x is x_(1/3)
  Vector<double> xc(Ac.Cols());
  CG(Ac, xc, b, rc, 1e-12, false); // Pass b for placeholder.
  x.Add(P.Mult(xc));
  x.Add(A.GaussSeidel(b - A.Mult(x))); // x is x_(2/3)

  return x;
}

int ML(const SparseMatrix<double>& A0,
       Vector<double>& x,
       const Vector<double>& x0,
       const Vector<double>& c,
       Vector<double>(*precond)
       (const std::vector<SparseMatrix<int>>&,
        const std::vector<SparseMatrix<double>>&,
        const Vector<double>&,
        const int,
        int k),
        int ncoarse,
        int max_level,
        bool verbose,
        double tol)
{
  std::vector<SparseMatrix<double>> A;
  A.push_back(A0);

  const int n = A0.Cols();
  double q = fmin(0.6, pow(1.0 * ncoarse / n, 1.0 / max_level));

  std::vector<SparseMatrix<int>> P;

  int L = GetSequence(P, A, ncoarse, q);
  printf("q = %.2f, L = %d, ", q, L);

  x = 0; // Set initial iterate to zero.
  // Because x = 0, the first residual r = c - A(x) = c. 
  Vector<double> r(c);
  
  Vector<double> y = (*precond)(P, A, r, L, 0); // Preconditioned residual.

  Vector<double> p(y); // Initial search direction.

  Vector<double> g; // See usage below.

  int num_iter = 0;
  double d0 = r.Mult(y); // r dot y
  Vector<double> d(A0.Cols() + 2); // Squares of residual norm.
  d[0] = d0;

  double alpha, beta, d1, t;

  // Beginning of PCG algorithm.
  for (int i = 0; i < A0.Rows() + 1; ++i)
  {
    A0.Mult(p, g); // g := A0p.
    t = p.Mult(g);
    alpha = d[i] / t;

    x.Add(alpha, p);
    r.Sub(alpha, g);

    // Copy two vectors by value. May be inefficient.
    y = (*precond)(P, A, r, L, 0);

    d1 = d[i];
    d[i + 1]  = r.Mult(y);

    ++num_iter;
    if (d[i + 1] < tol * tol * d0)
      break;

    beta = d[i + 1] / d1;
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
      printf("|r| = %.3e\n", sqrt(d[i]));

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

int GetSequence(std::vector<SparseMatrix<int>>& P,
                std::vector<SparseMatrix<double>>& A,
		            int ncoarse,
            		double q,
                int k)
{
  int nparts = std::max(2.0, A[k].Cols() * q); // METIS does not like nparts = 1.
  P.push_back(Unweighted(Partition(A[k], nparts)));
  A.push_back(P[k].Transpose().Mult(A[k].Mult(P[k])));
  A[k + 1].EliminateZeros();
  if (A[k].Cols() > ncoarse)
    k = GetSequence(P, A, ncoarse, q, k + 1);
  return k;
}

Vector<double> Multilevel(const std::vector<SparseMatrix<int>>& P,
                          const std::vector<SparseMatrix<double>>& A,
                          const Vector<double>& b,
                          const int L,
                          int k) // Current level

{ 
  Vector<double> x(A[k].ForwardGauss(b)); 
  Vector<double> r(P[k].MultAT(b - A[k].Mult(x))); // r_{k + 1}
  Vector<double> y(A[k + 1].Cols()); // x_{k + 1}
  if (L == k + 1)
    CG(A[L], y, b, r, 1e-12, false); // Pass b for placeholder.
  else
    y = Multilevel(P, A, r, L, k + 1);

  x.Add(P[k].Mult(y));
  x.Add(A[k].BackwardGauss(b - A[k].Mult(x))); 
  return x;
}
