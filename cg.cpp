#include <iostream>

#include "vectorview.hpp"
#include "parser.hpp"

using namespace linalgcpp;

double kernel(int x, int y, int n);

int main()
{
  // Read Coo list from text file  into sparse matrix.
  std::string filename;
  std::cout << "Please type the input filename: ";
  getline (std::cin, filename);
  SparseMatrix<double> A = ReadCooList(filename, true);
  
  /*
  int m = 9;
  int l = 3;

  CooMatrix<double> B(m);

  for (int i = 0; i < m; ++i)
  {		
    for (int j = 0; j < m; ++j)
      B.Add(i, j, kernel(i + 1, j + 1, l));
  }
 
  B.Print();

  SparseMatrix<double> A = B.ToSparse();
  */

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
  Vector<double> g; // g = Ap. (See below)
  
  double epsilon = 1e-10;
  // std::cout << "Tolerance = ";
  // std:: cin >> epsilon;

  int k = 0; // Number of iterations completed.
  double c0 = r.Mult(r); // (r^T)r
  double c = c0;

  double alpha, beta, c1, t;

  x = 0; // Initial iterate.

  // Beginning of CG algorithm.
  for (int i = 0; i < 100 * n; ++i)
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

  if (k < 100 * n)
  {
    std::cout << "The CG algorithm converged after " << k 
              << " iterations." << std::endl;
  }

  std::vector<double> diag = A.GetDiag();
  // diag = A.l1(diag);
  Vector<double> jacobi(diag); // Convert std::vector to linalgcpp::Vector.

  Vector<double> y; // Preconditioned residual. 
  y = r;
  y /= jacobi;

  p = y; // Initial search direction.
  c0 = r.Mult(y); // (r^T)y
  c = c0;
  x = 0; // Initial iterate.

  // Beginning of PCG algorithm.
  for (int i = 0; i < 1000; ++i)
  {
    g = A.Mult(p); // No overload for A * p.
    t = p.Mult(g);
    alpha = c / t;
    x += alpha * p;
    r -= alpha * g;
    y = r;
    y /= jacobi;
    c1 = c;
    c = r.Mult(y);

    ++k;
    if (c < epsilon * epsilon)
      break;

    beta = c / c1;
    p = y + beta * p;
  }

  if (k < 100 * n)
  {
    std::cout << "The PCG algorithm converged after " << k 
              << " iterations." << std::endl;
  }
  return 0;
}


double kernel(int x, int y, int n)
{
  double temp = n;
  double step = (1.0 / (temp - 1));

  double y1, y2, x1, x2;
  double counter = 0.0;

  y1 = ((x - 1) % n);

  for(int i = 0; i < n; ++i)
  {
    if (x - (n * i) > 0)
      counter = i;
  }

  x1 = counter;

  y2 = ((y - 1) % n);
	
  for (int i = 0; i < n; ++i)	
  {
    if (y - (n * i) > 0)
      counter = i;	
  }

  x2 = counter;

  y1 = y1 * step;
  y2 = y2 * step;
  x1 = x1 * step;
  x2 = x2 * step;

  double alpha = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));

  double to_return = exp(-1.0 * alpha);
  return to_return; 
}