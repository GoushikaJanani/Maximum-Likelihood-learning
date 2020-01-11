This is an implementation of Multivariate Gaussian distribution to distinguish face and non-face images.
Each pixel in an image is assumed to be a random variable following Gaussian distribution. Another assumption is that those pixels are IID variables(independent and identically distributed variables).

Header-only Eigen library is used for matrix calculations. The multivariate gaussian distribution described below is implemented using cholesky decomposition.

<p>
    <img src= ./distribution.JPG >
    <br>
    <em>Source: Computer Vision: models, learning and inference</em>
</p>
