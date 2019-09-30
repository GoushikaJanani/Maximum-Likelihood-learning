#include "includes.h"
#include "Norm.h"

using namespace std;
using namespace cv;
using namespace Eigen;

/**
 *  This function reads images from the specified path and convert into a matrix
 *  such that number of images form rows
 *  and number of pixels form the columns

 */
MatrixXi read_images(String path, uint8_t img_size)
{
	vector<String> path_names;
	glob(path, path_names, true);
	int n_of_files = path_names.size();
	int n_of_pixels = img_size * img_size;
	MatrixXi data = MatrixXi::Random(n_of_files, n_of_pixels);
	Mat img;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
	for (int iter = 0; iter < n_of_files; iter++)
	{
		img = imread(path_names[iter], CV_LOAD_IMAGE_GRAYSCALE);
		if (img.empty())continue;
		cv::resize(img, img, Size(img_size, img_size));
		/* Converting cv Mat to Eigen*/
		cv::cv2eigen(img, eigen_mat);
		/* Flattening Eigen image*/
		Map<RowVectorXi> flat_eigen_mat(eigen_mat.data(), eigen_mat.size());
		/* Store flattened image in data container*/
		data.row(iter) = flat_eigen_mat;
	}

	/* Remove heap memory*/
	vector<String>().swap(path_names);
	return data;
}

int main()
{
	String face_path = "../../../data/sample/Face/*.jpg";
	uint8_t img_size = 10;
	MatrixXi face_data = read_images(face_path, img_size);
	/* Face prob dis*/
	Norm face_distribution(face_data);
	face_distribution.visualize();
	face_distribution.train();
	Eigen::MatrixXi test_img = (read_images("../../../data/Test Data/Face/face1235.jpg", img_size));
	double prob = face_distribution.test(test_img);
	std::cout << "log Prob = " << std::log(prob) << std::endl;
}