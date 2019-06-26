#include <iostream>

#include "vectorview.hpp"
#include "parser.hpp"

using namespace linalgcpp;

int main()
{
  // Read Coo list from text file  into sparse matrix.
  std::string filename;
  std::cout << "Please type the input filename: ";
  getline (std::cin, filename);
  SparseMatrix<double> A = ReadCooList(filename, true);

  // Let n be length of a vector.
  const int n = A.Rows();

  std::cout << "\nThe matrix is " << n << "x" << n 
            << "." <<std::endl;

  Vector<double> x(n);
  Randomize(x,-10.0, 10.0); // Fill x with random values.

  // We let b = Ax. Because x0 = 0, 
  // the first residual r = b - A(x0) = b. 
  Vector<double> r = A.Mult(x);
  Vector<double> p = r;
  Vector<double> g; // g = Ap. (See below.)
  
  double epsilon;
  std::cout << "Tolerance = ";
  std:: cin >> epsilon;

  int k = 0; // Number of iterations completed.
  double c0 = r.Mult(r); // (r^T)r
  double c = c0;

  double alpha, beta, c1, t;

  x = 0; // Initial iterate.

  // Beginning of CG algorithm.
  for (int i = 0; i < n; ++i)
  {
    g = A.Mult(p); // No overload for A * p.
    t = p.Mult(g);
    alpha = c / t;
    x += alpha * p;
    r -= alpha * g;
    c1 = c;
    c = r.Mult(r);

    ++k;
    if (c < epsilon * epsilon)
      break;

    beta = c / c1;
    p = r + beta * p;
  }

  if (k == n)
  {  
    std::cout << "The CG algorithm did not converge within " 
              << n << " iterations." << std::endl;
  }
  else
  {
    std::cout << "The CG algorithm converged after " << k 
              << " iterations." << std::endl;
  }

  return 0;
}
