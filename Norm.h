//#pragma once
//#include "calculateMetrics.h"
#include "projHeaders.h"
#define _USE_MATH_DEFINES
#include <math.h>

class Norm
{
private:
	int imageSize;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> data;
	Eigen::VectorXi mean;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> covariance;
	double rt2pi;
	double sqrtDetCovar;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> covarianceInverse;

public:


	void train();
	double test(Eigen::MatrixXi& );
	

	Norm(Eigen::MatrixXi Imgs);
	Eigen::MatrixXd useOnlyDiagElements(Eigen::MatrixXd&);

	void visualize();
};

