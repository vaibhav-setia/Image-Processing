
#include <iostream>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <stdio.h>

#define SIGMA_CLIP 6.0f
using namespace cv;
using namespace std;
 int r=2;
void updateResult(Mat complex)
{
    Mat work;
    idft(complex, work);
    Mat planes[] = {Mat::zeros(complex.size(), CV_32F), Mat::zeros(complex.size(), CV_32F)};
    split(work, planes);                // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
 
    magnitude(planes[0], planes[1], work);    // === sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
    normalize(work, work, 0, 1, NORM_MINMAX);
    imshow("result", work);
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
Mat createGausFilterMask(Size imsize, int radius) {
    
	// call openCV gaussian kernel generator
	double sigma = (r/SIGMA_CLIP+0.5f);
	Mat kernelX = getGaussianKernel(2*radius+1, sigma, CV_32F);
	Mat kernelY = getGaussianKernel(2*radius+1, sigma, CV_32F);
	// create 2d gaus
	Mat kernel = kernelX * kernelY.t();
//Mat kernel = (Mat_<float>(3, 3) << 0.111,  0.111,  0.111,
  //                               0.111,  0.111,  0.111,
//				 0.111,  0.111,  0.111);

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

//code reference https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
int main( int argc, char** argv )
{ 
 
 String file;
    file = "lena.png";
 
    Mat image = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
imshow("inout image", image);

 Mat padded;                            	
						
    int m = getOptimalDFTSize( image.rows );
    int n = getOptimalDFTSize( image.cols );  
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n -image.cols, BORDER_CONSTANT, Scalar::all(0));//expand input image to optimal size , on the border add zero values

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);  
dft(complexI, complexI); //computing dft
//split(complexI, planes); //image converted to complex and real dft here


 

   Mat mask = createGausFilterMask(padded.size(),r );  
shift(mask);// Forming the gaussian filter
Mat mplane[] = {Mat_<float>(mask), Mat::zeros(mask.size(), CV_32F)};
Mat kernelcomplex;
    merge(mplane, 2, kernelcomplex); 

dft(kernelcomplex, kernelcomplex);

split(kernelcomplex, mplane);// splitting the dft of kernel to real and complex 
mplane[1]=mplane[0];
Mat kernel_spec;
merge(mplane, 2, kernel_spec);
mulSpectrums(complexI, kernel_spec, complexI, DFT_ROWS);
/*
Mat xplanes[] = {Mat::zeros(complexI.size(), CV_32F), Mat::zeros(complexI.size(), CV_32F)};
    split(complexI, xplanes); 
magnitude(xplanes[0], xplanes[1], xplanes[0]);

mulSpectrums(planes[0], mplane[0], planes[0], DFT_ROWS);//  multiplying real elements

mulSpectrums(planes[1], mplane[0], planes[1], DFT_ROWS);// multiplying imaginary elements

//magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = xplanes[0]; // merging both real and imaginary planes

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
*/
Mat magI=updateMag(complexI);
   // imshow("Input Image"       , I   );    // Show the result
    imshow("spectrum magnitude", magI);

/*Mat temp=updateMag(kernel_spec);  // computing magnitude

namedWindow( "image fourier", CV_WINDOW_AUTOSIZE );

imshow("image fourier", temp);*/

updateResult(complexI); //converting to viewable form, computing idft 
/* Mat inverseTransform;
    idft(magI, inverseTransform, DFT_INVERSE|DFT_REAL_OUTPUT);
    normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
    imshow("Reconstructed", inverseTransform);*/

  waitKey(0); 

return 0;
}
