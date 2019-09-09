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
        int max_level = 8,
        bool verbose = false,
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

  int level;
  std::cout << "Enter the number of levels: ";
  std::cin >> level;
  
  for (int k = 0; k < 10; ++k)
  {
    ML(A, x, x0, b, Multilevel, cbrt(n), level);
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
  double q = pow(1.0 * ncoarse / n, 1.0 / max_level);
  // double q = fmin(0.6, pow(1.0 * ncoarse / n, 1.0 / max_level));

  std::vector<SparseMatrix<int>> P;

  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  duration<double> duration;

  t1 = high_resolution_clock::now();
  int L = GetSequence(P, A, ncoarse, q);
  t2 = high_resolution_clock::now();
  printf("%.5f %d ", q, L);
  duration = t2 - t1;
  printf("%.6f ", duration.count());
  
  t1 = high_resolution_clock::now();
  x = 0; // Set initial iterate to zero.
  // Because x = 0, the first residual r = c - A(x) = c. 
  Vector<double> r(c);
  
  Vector<double> y = (*precond)(P, A, r, L, 0); // Preconditioned residual.

  Vector<double> p(y); // Initial search direction.

  Vector<double> g; // See usage below.

  int num_iter = 0;
  double d0 = r.Mult(y); // r dot y
  double  d(d0);

  double alpha, beta, d1, t;

  // Beginning of PCG algorithm.
  for (int i = 0; i < A0.Rows() + 1; ++i)
  {
    A0.Mult(p, g); // g := A0p.
    t = p.Mult(g);
    alpha = d / t;

    x.Add(alpha, p);
    r.Sub(alpha, g);

    // Copy two vectors by value. May be inefficient.
    y = (*precond)(P, A, r, L, 0);

    d1 = d;
    d = r.Mult(y);

    ++num_iter;
    if (d < tol * tol * d0)
      break;

    beta = d / d1;
    p *= beta;
    p += y;
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
  t2 = high_resolution_clock::now();
  duration = t2 - t1;
  r = x - x0;
  double error(L2Norm(r));
  printf("%.6f %d %.3e 0\n", duration.count(), num_iter, error);
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
  Vector<double> x(A[k].Jacobi(b)); 
  Vector<double> r(P[k].MultAT(b - A[k].Mult(x))); // r_{k + 1}
  Vector<double> y(A[k + 1].Cols()); // x_{k + 1}
  if (L == k + 1)
    CG(A[L], y, b, r, 1e-12, false); // Pass b for placeholder.
  else
    y = Multilevel(P, A, r, L, k + 1);

  x.Add(P[k].Mult(y));
  x.Add(A[k].Jacobi(b - A[k].Mult(x))); 
  return x;
}
