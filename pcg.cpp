#include "vectorview.hpp"
#include "parser.hpp"

using namespace linalgcpp;

// Preconditioned conjugate gradient method.
// Returns number of iterations.
int myPCG(const SparseMatrix<double>& A,
           Vector<double>& x,
           const Vector<double>& b,
           Vector<double>(SparseMatrix<double>::*precond)
           (Vector<double>) const,
           double tol = 1e-16,
	   bool verbose = true);

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

  // Each preconditioner is a method of the SparseMatrix
  // class. The syntax leaves something to be desired.
  std::cout << "Jacobi: ";
  myPCG(A, x, b, &SparseMatrix<double>::Jacobi);

  std::cout << "l1-smoother: ";
  myPCG(A, x, b, &SparseMatrix<double>::L1);

  std::cout << "Gauss-Seidel: ";
  myPCG(A, x, b, &SparseMatrix<double>::GaussSeidel);

  return 0;
}

// This function solves Ax = b using one of the three
// preconditioners shown above.
int myPCG(const SparseMatrix<double>& A,
           Vector<double>& x,
           const Vector<double>& b,
           Vector<double>(SparseMatrix<double>::*precond)
           (Vector<double>) const,
           double tol,
	   bool verbose)
{ 
  x = 0; // Set initial iterate to zero.
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
    if (c[i + 1] < tol * tol)
      break;

    beta = c[i + 1] / c1;
    p *= beta;
    p += y;
  }
  // If num_iter > A.Cols(), then the algorithm
  // did not converge in the expected (theoretical)
  // number of iterations.
  printf("num_iter = %d\n", num_iter);

  if (verbose)
  {
    // Print the last three squares of the residual 
    // norm that are computed.
    // for (int i = num_iter - 2; i < num_iter + 1; ++i)
    //   printf("c[%d] = %.3e\n", i, c[i]);

    // Print the last three residual norms 
    // that are computed.
    for (int i = num_iter - 2; i < num_iter + 1; ++i)
      printf("|r| = %.3e\n", sqrt(c[i]));

    // Let us see how close the approximation is
    // in the euclidean norm.
    r = b - A.Mult(x);
    double error(L2Norm(r));

    std::cout << "Compare the approx soln with the exact: ";
    printf("|x - x0| = %.3e\n", error);
    std::cout << std::endl;
  }
  return num_iter;
}