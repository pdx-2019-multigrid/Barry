#include <iostream>

#include "vectorview.hpp"
#include "parser.hpp"

using namespace linalgcpp;

// Solve Ax = b using a preconditioned conjugate gradient method.
int myPCG(const SparseMatrix<double>& A,
           Vector<double>& x,
           const Vector<double>& b,
           Vector<double>(SparseMatrix<double>::*precond)
           (Vector<double>) const,
           double tol = 1e-16,
	   bool verbose = true);

int main()
{
  // Create graph laplacian from edge list.
  std::string filename;
  std::cout << "Please type the input filename: ";
  getline(std::cin, filename);
  SparseMatrix<double> A(ReadGraphList(filename));

  // Make the matrix positive definite.
  A.EliminateRowCol(A.Rows() - 1);

  // Let n be length of a vector.
  const int n = A.Cols();

  // Compare size of matrix with number of iterations
  // for (preconditioned) conjugate gradient.
  std::cout << "The adjacency matrix is " << n << "x" << n 
            << ".\n" <<std::endl;

  Vector<double> x0(n); // Exact solution.
  Randomize(x0, -10.0, 10.0); // Fill x0 with random values.
  Vector<double> b(A.Mult(x0)); // Definition of b.

  Vector<double> x(n); // Iterate x.
  int num_iter;

  std::cout << "None: ";
  num_iter = myPCG(A, x, b, &SparseMatrix<double>::None);

  std::cout << "Jacobi: ";
  myPCG(A, x, b, &SparseMatrix<double>::Jacobi);

  std::cout << "l1-smoother: ";
  myPCG(A, x, b, &SparseMatrix<double>::L1);

  std::cout << "Gauss-Seidel: ";
  myPCG(A, x, b, &SparseMatrix<double>::GaussSeidel);

  return 0;
}

int myPCG(const SparseMatrix<double>& A,
           Vector<double>& x,
           const Vector<double>& b,
           Vector<double>(SparseMatrix<double>::*precond)
           (Vector<double>) const,
           double tol,
	   bool verbose)
{ 
  x = 0; // Set initial iterate to zero.
  // We let b = Ax. Because x = 0, 
  // the first residual r = b - A(x) = b. 
  Vector<double> r(b);
  Vector<double> y = (A.*precond)(r); // Preconditioned residual.

  Vector<double> p(y); // Initial search direction.

  Vector<double> g; // See usage below.

  int num_iter = 0;
  double c0 = r.Mult(y); // (r^T)y
  Vector<double> c(A.Cols() + 11); // Squares of residual norm
  c[0] = c0;

  double alpha, beta, c1, t;

  // Beginning of PCG algorithm.
  for (int i = 0; i < A.Rows() + 10; ++i)
  {
    A.Mult(p, g); // g := Ap.
    t = p.Mult(g);
    alpha = c[i] / t;

    x.Add(alpha, p);
    r.Sub(alpha, g);

    y = (A.*precond)(r);

    c1 = c[i];
    c[i + 1]  = r.Mult(y);

    ++num_iter;
    if (c[i + 1] < tol * tol)
      break;

    beta = c[i + 1] / c1;
    p *= beta;
    p += y;
  }
  printf("num_iter = %d\n", num_iter);
  if (verbose)
  {
    for (int i = num_iter - 2; i < num_iter + 1; ++i)
      printf("|r|^2[%d] = %.3e\n", i, c[i]);

    r = b - A.Mult(x);
    double error(L2Norm(r));

    std::cout << "Compare the approx soln with the exact: ";
    printf("|x - x0| = %.3e\n", error);
    std::cout << std::endl;
  }
  return num_iter;
}
