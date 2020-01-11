#include "Norm.h"


Norm::Norm(Eigen::MatrixXi& dataImg) : data(dataImg)
{
	img_size = (int)sqrt(data.cols());
}

/**
 *  This function calculates mean of every pixel from all the training images.
 *  Each pixel is assumed to be iid random variable. 
 *  Hence only the diagonal elements of covariance matrix are used
 */
void Norm::train()
{
	mean = (data.colwise().mean());
	
	Eigen::MatrixXi centred_img = (data.transpose().colwise() - mean).transpose();
	
	int n_of_files = data.rows();
	covariance= ((centred_img.transpose() * centred_img) / (n_of_files)).cast<double>();
	covariance = use_only_diag_elements(covariance);
	/* Checking if covariance is positive definite so that Cholesky decomposition can be applied*/
	Eigen::LLT<Eigen::MatrixXd>lltObj(covariance);
	if (lltObj.info() == Eigen::NumericalIssue)
	{
		throw std::runtime_error("Cholesky is not possible !");
	}
	//int n_var = data.cols(); /* Number of random variables*/
	//sqrt_det_covar = std::pow(covariance.determinant(), -0.5);
	//sqrt_det_covar = 1;
	covar_inverse = covariance.inverse();

}

/**
 *  This function uses the calculated mean and covariance from training images
 *  to find the logarithmic probability of an image from multivariate Gaussian distribution
 *  Cholesky decomposition is used to implement the distribution function.
 */
double Norm::test(Eigen:: MatrixXi &x)
{
	/*                     1
	/* f(x) = ------------------------ * exp(-0.5(x-u)^T * sigma^-1 * (x-u)
	/*        (2pi)^D/2 * |sigma|^1/2									*/

	//Below is direct implementation of formula. Did not work

	//Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> xMinusMu;
	//xMinusMu = ((x.transpose().colwise() - mean).transpose()).cast<double>();
	//double constantFactor = 1 / (rt_2pi * sqrt_det_covar);
	//double logprob = ((xMinusMu * covar_inverse * xMinusMu.transpose() )(0, 0) / -2) - log(rt_2pi) - log(sqrt_det_covar);
	//std::cout << "Prob = " << (logprob);

	//Applying Cholesky Decomposition
	const double log_sqrt_2Pi = 0.5 * std::log(2 * M_PI);
	typedef Eigen::LLT<Eigen::MatrixXd> Chol;
	Chol chol(covariance);
	// Handle non positive definite covariance somehow:
	if (chol.info() != Eigen::Success) throw "decomposition failed!";
	const Chol::Traits::MatrixL L_mat = chol.matrixL();
	Eigen::MatrixXd Sub = ((x.transpose().colwise() - mean)).cast<double>();
	double quadform = (L_mat.solve(Sub)).squaredNorm();
	return std::exp(-x.rows() * log_sqrt_2Pi - 0.5 * quadform) / L_mat.determinant();


}

void Norm::visualize()
{	
	train();
	cv::Mat mean_img_mat;
	eigen2cv(mean, mean_img_mat);
	/* Reshape meanFace to img_size * img_size */
	cv::Mat reshaped_mean_img = mean_img_mat.reshape(1, img_size);
	reshaped_mean_img.convertTo(reshaped_mean_img, CV_8UC1);
	cv::namedWindow("MeanImg", CV_WINDOW_NORMAL);
	imshow("MeanImg", reshaped_mean_img);
	cv::waitKey(0);
	cv::destroyAllWindows();
}


Eigen::MatrixXd Norm::use_only_diag_elements(Eigen::MatrixXd& mat)
{
	uint16_t rows, cols;
	rows = mat.rows();
	cols = mat.cols();
	Eigen::MatrixXd zero_mat = Eigen::MatrixXd::Zero(rows, cols);
	for (int iRow = 0; iRow < rows; iRow++)
	{
		for (int jCol = 0; jCol < cols; jCol++)
		{
			if (iRow == jCol)
			{
				zero_mat(iRow, jCol) = mat.coeff(iRow, jCol);
				break;
			}
		}
	}

	return zero_mat;
}