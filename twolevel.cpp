#include "partition.hpp"
#include <chrono>

using namespace linalgcpp;
using namespace std::chrono;

// Regular conjugate gradient method.
// Returns number of iterations.
int CG(const SparseMatrix<double>& A,
       Vector<double>& x,
       const Vector<double>& x0,
       const Vector<double>& b,
       double tol = 1e-9,
       bool verbose = false);

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
       bool verbose = false);

// Two-level preconditioner used in function TL.
Vector<double> TwoLevel(const SparseMatrix<double>& A, 
                        const Vector<double>& b,
                        const SparseMatrix<int>& P,
                        const SparseMatrix<double>& Ac);

int main()
{
  // Create graph Laplacian from edge list.
  // Must have zeroth vertex.
  std::string graph;
  std::cout << "Please type the graph filename without extension: ";
  getline(std::cin, graph);
  std::string full("../data/" + graph + ".txt");
  SparseMatrix<double> A(ReadGraphList(full));

  // Make the matrix positive definite.
  A.EliminateRowCol(A.Rows() - 1);
  A.EliminateZeros();

  // Let n be length of a vector.
  const int n = A.Cols();

  // Compare size of matrix with number of iterations
  std::cout << "The matrix is " << n << "x" << n 
            << std::endl;

  std::string vector("../data/v" + graph + ".txt");
  Vector<double> x0(ReadText(vector)); // Exact solution x0.
  Vector<double> b(A.Mult(x0)); // Definition of b.

  Vector<double> x(n); // Iterate x.

  for (int k = 0; k < 10; ++k)
  {
    TL(A, x, x0, b, TwoLevel);
  }
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
  double c(c0);

  double alpha, beta, c1, t;

  // Beginning of CG algorithm.
  for (int i = 0; i < A.Rows() + 1; ++i)
  {
    A.Mult(p, g); // g := Ap.
    t = p.Mult(g);
    alpha = c / t;

    x.Add(alpha, p);
    r.Sub(alpha, g);

    c1 = c;
    c = r.Mult(r);

    ++num_iter;
    if (c < tol * tol * c0)
      break;

    beta = c / c1;
    p *= beta;
    p += r;
  }

  if (verbose)
  {
    // If num_iter > A.Cols(), then the algorithm
    // did not converge in the expected (theoretical)
    // number of iterations.
    printf("num_iter = %d\n", num_iter);   
 
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
  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  duration<double> duration;

  t1 = high_resolution_clock::now();
  // Determine interpolation matrix P and coarse graph Laplacian Ac.
  int nparts = std::max(2.0, cbrt(A.Cols()));
  SparseMatrix<int> P(Unweighted(Partition(A, nparts)));
  SparseMatrix<double> Ac = P.Transpose().Mult(A.Mult(P));
  Ac.EliminateZeros();
  t2 = high_resolution_clock::now();
  duration = t2 - t1;
  printf("%.6f ", duration.count());

  t1 = high_resolution_clock::now();
  x = 0; // Set initial iterate to zero.
  // Because x = 0, the first residual r = b - A(x) = b. 
  Vector<double> r(b);
  Vector<double> y = (*precond)(A, r, P, Ac); // Preconditioned residual.
  Vector<double> p(y); // Initial search direction.
  Vector<double> g; // See usage below.
  int num_iter = 0;
  double c0 = r.Mult(y); // r dot y
  double c(c0);

  double alpha, beta, c1, t;

  // Beginning of PCG algorithm.
  for (int i = 0; i < A.Rows() + 1; ++i)
  {
    A.Mult(p, g); // g := Ap.
    t = p.Mult(g);
    alpha = c / t;

    x.Add(alpha, p);
    r.Sub(alpha, g);

    // Copy two vectors by value. May be inefficient.
    y = (*precond)(A, r, P, Ac);

    c1 = c;
    c = r.Mult(y);

    ++num_iter;
    if (c < tol * tol * c0)
      break;

    beta = c / c1;
    p *= beta;
    p += y;
  }

  if (verbose)
  {
    // If num_iter > A.Cols(), then the algorithm
    // did not converge in the expected (theoretical)
    // number of iterations.
    printf("nparts = %d, num_iter = %d\n", nparts, num_iter);

    // Let us see how close the approximation is
    // in the euclidean norm.
    r = x - x0;
    double error(L2Norm(r));

    std::cout << "Compare the approx soln with the exact: ";
    printf("|x - x0| = %.3e\n", error);
    std::cout << std::endl;
  }
  t2 = high_resolution_clock::now();
  r = x - x0;
  double error(L2Norm(r));
  duration = t2 - t1;
  printf("%.6f %d %.3e 0\n", duration.count(), num_iter, error);
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