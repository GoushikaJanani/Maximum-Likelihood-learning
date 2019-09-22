#include "projHeaders.h"
#include "Norm.h"

using namespace std;
using namespace cv;
using namespace Eigen;

MatrixXi readImages(String path, uint8_t imgSize)
{
	vector<String> pathNames;
	glob(path, pathNames, true);
	int numOfFiles = pathNames.size();
	int numOfPixels = imgSize * imgSize;
	MatrixXi data = MatrixXi::Random(numOfFiles, numOfPixels);
	Mat img;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> eigenMat;
	for (int iter = 0; iter < numOfFiles; iter++)
	{
		img = imread(pathNames[iter], CV_LOAD_IMAGE_GRAYSCALE);
		if (img.empty())continue;
		cv::resize(img, img, Size(imgSize, imgSize));
		/* Converting cv Mat to Eigen*/
		cv::cv2eigen(img, eigenMat);
		/* Flattening Eigen image*/
		Map<RowVectorXi> flattenedEigenMat(eigenMat.data(), eigenMat.size());
		/* Store flattened image in data container*/
		data.row(iter) = flattenedEigenMat;
	}

	/* Remove heap memory*/
	vector<String>().swap(pathNames);
	return data;
}

// TO DO: Remove
//Pass by reference
Eigen::MatrixXd DiagElements(Eigen::MatrixXd& mat)
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
	//ZeroMat is dynamically allocated. So it is safe to return
	return zeroMat;
}
int main()
{
	String facePath = "../../../data/sample/Face/*.jpg";
	uint8_t imageSize = 10;
	MatrixXi faceData = readImages(facePath, imageSize);
	//cout << "Rows = " << faceData.rows() << "\t Cols = " << faceData.cols() << endl;

	/* Face prob dis*/
	Norm faceDis(faceData);
	faceDis.train();
	Eigen::MatrixXi testImage = (readImages("../../../data/Test Data/Face/face1235.jpg", imageSize));
	double prob = faceDis.test(testImage);
	std::cout << "log Prob = " << std::log(prob) << std::endl;

	/*Eigen::MatrixXd mat(2, 2);
	mat(0, 0) = 1;
	mat(0, 1) = 2;
	mat(1, 0) = 3;
	mat(1, 1) = 4;
	Eigen::MatrixXd res = DiagElements(mat);
	std::cout << res << std::endl;*/
}