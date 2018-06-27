#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <stdio.h>
#define SIGMA_CLIP 6.0f
using namespace cv;
using namespace std; 

Mat updateMag(Mat complex);
void updateResult(Mat complex);
 
Mat computeDFT(Mat image);
Mat createGausFilterMask(Size imsize, int radius);
void shift(Mat magI);
 
int kernel_size = 0;
int r = 100; 
int main( int argc, char** argv )
{ 
 
    String file;
    file = "lena.png";
 
    Mat image = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
    namedWindow( "Orginal window", CV_WINDOW_AUTOSIZE  );// Create a window for display.
    imshow( "Orginal window", image );                   // Show our image inside it.
 
    Mat complex = computeDFT(image);
	/*Mat temp=updateMag(complex); 
namedWindow( "image fourier", CV_WINDOW_AUTOSIZE );
imshow("image fourier", temp);*/
 
    namedWindow( "spectrum", CV_WINDOW_AUTOSIZE );

   
   Mat mask = createGausFilterMask(complex.size(),r );

shift(mask);
//mask= computeDFT(mask);  //Compute DFT of mask
//mask =updateMag(mask);   //show the mask spectrum
imshow("gaus-mask", mask);
 Mat planes[] = {Mat::zeros(complex.size(), CV_32F), Mat::zeros(complex.size(), CV_32F)};
    Mat kernel_spec;
    planes[0] = mask; // real
    planes[1] = mask; // imaginar
    merge(planes, 2, kernel_spec);
 
    mulSpectrums(complex, kernel_spec, complex, DFT_ROWS);
  Mat temp = updateMag(complex); 
imshow("spectrum", temp);
        // compute magnitude of complex, switch to logarithmic scale and display...
    updateResult(complex);      // do inverse transform and display the result image
    waitKey(0); 
 
    return 0;
}
 

void updateResult(Mat complex)
{
    Mat work;
    idft(complex, work);
//  dft(complex, work, DFT_INVERSE + DFT_SCALE);
    Mat planes[] = {Mat::zeros(complex.size(), CV_32F), Mat::zeros(complex.size(), CV_32F)};
    split(work, planes);                // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
 
    magnitude(planes[0], planes[1], work);    // === sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
    normalize(work, work, 0, 1, NORM_MINMAX);
    imshow("result", work);
}
 
Mat updateMag(Mat complex )
{
 
    Mat magI;
    Mat planes[] = {Mat::zeros(complex.size(), CV_32F), Mat::zeros(complex.size(), CV_32F)};
    split(complex, planes);                // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
 
    magnitude(planes[0], planes[1], magI);    // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
 
    // switch to logarithmic scale: log(1 + magnitude)
    magI += Scalar::all(1);
    log(magI, magI);
 
    shift(magI);
    normalize(magI, magI, 1, 0, NORM_INF); // Transform the matrix with float values into a
             return magI;                                 // viewable image form (float between values 0 and 1).
    //imshow("spectrum", magI);
}
 

 
Mat computeDFT(Mat image) {
    
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( image.rows );
    int n = getOptimalDFTSize( image.cols ); // on the border add zero values
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complex;
    merge(planes, 2, complex);         // Add to the expanded another plane with zeros
    dft(complex, complex, DFT_COMPLEX_OUTPUT);  // furier transform
    return complex;
}
 
Mat createGausFilterMask(Size imsize, int radius) {
    
	// call openCV gaussian kernel generator
	double sigma = (r/SIGMA_CLIP+0.5f);
	Mat kernelX = getGaussianKernel(2*radius+1, sigma, CV_32F);
	Mat kernelY = getGaussianKernel(2*radius+1, sigma, CV_32F);
	// create 2d gaus
Mat kernel = kernelX * kernelY.t();
/*Mat kernel = (Mat_<float>(3, 3) << 0.111,  0.111,  0.111,
                                0.111,  0.111,  0.111,
			 0.111,  0.111,  0.111);*/

	int w = imsize.width-kernel.cols;
	int h = imsize.height-kernel.rows;

	int r = w/2;
	int l = imsize.width-kernel.cols -r;

	int b = h/2;
	int t = imsize.height-kernel.rows -b;

	Mat ret;
	copyMakeBorder(kernel,ret,t,b,l,r,BORDER_CONSTANT,Scalar::all(0));

	return ret;
    
}
 
void shift(Mat magI) {
 
    // crop if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
 
    int cx = magI.cols/2;
    int cy = magI.rows/2;
 
    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
 
    Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
