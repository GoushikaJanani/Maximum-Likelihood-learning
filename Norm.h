//#pragma once
//#include "calculateMetrics.h"
#include "includes.h"
#define _USE_MATH_DEFINES
#include <math.h>

class Norm
{
private:
	int img_size;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> data;
	Eigen::VectorXi mean;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> covariance;
	//double rt_2pi;
	//double sqrt_det_covar;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> covar_inverse;

public:
	void train();
	double test(Eigen::MatrixXi& );
	Norm(Eigen::MatrixXi Imgs);
	Eigen::MatrixXd use_only_diag_elements(Eigen::MatrixXd&);
	void visualize();
};

