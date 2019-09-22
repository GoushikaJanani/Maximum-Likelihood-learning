#include "Norm.h"


Norm::Norm(Eigen::MatrixXi dataImg) : data(dataImg)
{
	/*data = dataImg;*/
	imageSize = (int)sqrt(data.cols());
}
void Norm::train()
{
	mean = (data.colwise().mean());
	
	Eigen::MatrixXi centredImg = (data.transpose().colwise() - mean).transpose();
	
	int numOfFiles = data.rows();
	covariance= ((centredImg.transpose() * centredImg) / (numOfFiles)).cast<double>();
	covariance = useOnlyDiagElements(covariance);




	/* Checking if covariance is positive definite so that Cholesky decomposition can be applied*/
	Eigen::LLT<Eigen::MatrixXd>lltObj(covariance);
	if (lltObj.info() == Eigen::NumericalIssue)
	{
		throw std::runtime_error("Cholesky is not possible !");
	}
	//int n_var = data.cols(); /* Number of random variables*/
	//sqrtDetCovar = std::pow(covariance.determinant(), -0.5);
	//sqrtDetCovar = 1;
	
	covarianceInverse = covariance.inverse();

}

double Norm::test(Eigen:: MatrixXi &x)
{
	/*                     1
	/* f(x) = ------------------------ * exp(-0.5(x-u)^T * sigma^-1 * (x-u)
	/*        (2pi)^D/2 * |sigma|^1/2									*/

	//Below is direct implementation of formula. Did not work

	//Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> xMinusMu;
	//xMinusMu = ((x.transpose().colwise() - mean).transpose()).cast<double>();
	//double constantFactor = 1 / (rt2pi * sqrtDetCovar);

	//double logprob = ((xMinusMu * covarianceInverse * xMinusMu.transpose() )(0, 0) / -2) - log(rt2pi) - log(sqrtDetCovar);
	//std::cout << "Prob = " << (logprob);


	//Applying Cholesky Decomposition
	// avoid magic numbers in your code. Compilers will be able to compute this at compile time:
	
	const double logSqrt2Pi = 0.5 * std::log(2 * M_PI);
	typedef Eigen::LLT<Eigen::MatrixXd> Chol;
	Chol chol(covariance);
	// Handle non positive definite covariance somehow:
	if (chol.info() != Eigen::Success) throw "decomposition failed!";
	const Chol::Traits::MatrixL L = chol.matrixL();
	Eigen::MatrixXd Sub = ((x.transpose().colwise() - mean)).cast<double>();
	double quadform = (L.solve(Sub)).squaredNorm();
	return std::exp(-x.rows() * logSqrt2Pi - 0.5 * quadform) / L.determinant();


}

//double Norm::foo(Eigen::MatrixXi x)
//{
//	// avoid magic numbers in your code. Compilers will be able to compute this at compile time:
//	const double logSqrt2Pi = 0.5 * std::log(2 * M_PI);
//	typedef Eigen::LLT<Eigen::MatrixXd> Chol;
//	Chol chol(covariance);
//	// Handle non positive definite covariance somehow:
//	if (chol.info() != Eigen::Success) throw "decomposition failed!";
//	const Chol::Traits::MatrixL L = chol.matrixL();
//	double quadform = (L.solve(x - mean)).squaredNorm();
//	return std::exp(-x.rows() * logSqrt2Pi - 0.5 * quadform) / L.determinant();
//}
void Norm::visualize()
{
	
	train();
	cv::Mat matmeanImg;
	//eigen2cv(meanImg, matmeanImg);
	///* Reshape meanFace to imageSize * imageSize */
	//cv::Mat reshapedMeanImg = matmeanImg.reshape(1, imageSize);
	//reshapedMeanImg.convertTo(reshapedMeanImg, CV_8UC1);
	//cv::namedWindow("MeanImg", CV_WINDOW_NORMAL);
	//imshow("MeanImg", reshapedMeanImg);
	//cv::waitKey(0);
	//cv::destroyAllWindows();
}


Eigen::MatrixXd Norm::useOnlyDiagElements(Eigen::MatrixXd& mat)
{
	uint16_t rows, cols;
	rows = mat.rows();
	cols = mat.cols();
	Eigen::MatrixXd zeroMat = Eigen::MatrixXd::Zero(rows, cols);
	for (int iRow = 0; iRow < rows; iRow++)
	{
		for (int jCol = 0; jCol < cols; jCol++)
		{
			if (iRow == jCol)
			{
				zeroMat(iRow, jCol) = mat.coeff(iRow, jCol);
				break;
			}
		}
	}

	return zeroMat;
}