#include <Eigen/Dense>
#include <iostream>

//int main()
//{
//	Eigen::MatrixXd A(3, 3);
//	A << 4, -1, 2, -1, 6, 0, 2, 0, 5;
//	std::cout << "Matrix A is\n" << A << "\n";
//	Eigen::LLT<Eigen::MatrixXd>lltOf(A);
//	if (lltOf.info() == Eigen::NumericalIssue)
//	{
//		throw std::runtime_error("Possibly non positive definite matrix");
//	}
//	Eigen::MatrixXd L = lltOf.matrixL();
//	std::cout << "Matrix A is back is\n" << L*L.transpose() << "\n";
//
//}