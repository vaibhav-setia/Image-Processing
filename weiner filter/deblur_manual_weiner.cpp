//#include <boost/program_options.hpp>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
//namespace po = boost::program_options;

Mat updateResult(Mat complex)
{
    Mat work;
    idft(complex, work);
//  dft(complex, work, DFT_INVERSE + DFT_SCALE);
    Mat planes[] = {Mat::zeros(complex.size(), CV_32F), Mat::zeros(complex.size(), CV_32F)};
    split(work, planes);                // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
 
    magnitude(planes[0], planes[1], work);    // === sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
    normalize(work, work, 0, 1, NORM_MINMAX);
return work;
    imshow("result", work);
}

Mat wiener2(Mat I, Mat image_spectrum, int noise_stddev);
Mat padd_image(Mat I);

Mat get_spectrum(Mat I);
Mat get_dft(Mat I);

Mat with_noise(Mat image, int stddev);
Mat rand_noise(Mat I, int stddev);

Mat createavg(Size imsize) ;
void shift(Mat magI);


int main(int argc, char *argv[]) {

	int noise_stddev=5;
	string input_filename="blur.png", output_filename="write.png";   // Have a blurred image here

	cout << "noise standard deviation: " << noise_stddev << "\n";
	cout << "input file: " << input_filename << "\n";

	/*if (vm.count("help")) {
		cout << desc << "\n";
		return 1;
	}*/

	Mat I = imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);  //READ THE BLUR IMAGE
	if(I.data==NULL){
		cout << "Can't open file: " << input_filename << "\n";
		return 2;
	}

	Mat raw_sample = imread("sample.bmp", CV_LOAD_IMAGE_GRAYSCALE); // READ THE SAMPLE IMAGE WHOSE SPECTRUM IS EXPECTED TO BE SIMILAR TO THE IMAGE
	if(raw_sample.data==NULL){
		cout << "Can't open file: sample.bmp\n";
		return 3;
	}
imshow("initial blurred image", I);
	Mat padded = padd_image(I);
	Mat noisy;
	//if(vm.count("generate-noisy")){
		noisy = with_noise(padded, noise_stddev);
		//imwrite(output_filename, noisy);
		//return 0;
	//}else{
	//	noisy = padded;
	//}

	Mat sample(padded.rows, padded.cols, CV_8U);
	resize(raw_sample, sample, sample.size());    
	Mat spectrum = get_spectrum(sample);    //to get signal spectrum of known image 
	Mat enhanced = wiener2(noisy, spectrum, noise_stddev);

//fastNlMeansDenoising(enhanced,enhanced,20);  // Denoising the output final image again

	//imwrite(output_filename, enhanced);

	//if(vm.count("show")){
		imshow("image 1", noisy);
		imshow("image 2", enhanced);
	//}
	waitKey();
}
Mat createavg(Size imsize) {
    
	// call openCV gaussian kernel generator
	/*double sigma = (r/SIGMA_CLIP+0.5f);
	Mat kernelX = getGaussianKernel(2*radius+1, sigma, CV_32F);
	Mat kernelY = getGaussianKernel(2*radius+1, sigma, CV_32F);*/
//Mat kernel = Mat(5,5,CV_32FC1,Scalar(0.04));
//Mat kernel = imread("psf.jpg",CV_LOAD_IMAGE_GRAYSCALE);
Mat kernel = (Mat_<double>(5,5) <<  0.039723,0.039928,0.040126,0.040068,0.040175,
0.039709,0.039918,0.040115,0.04006,0.040164,
0.039705,0.039914,0.040185,0.040053,0.040159,
0.039712,0.039919,0.040117,0.040059,0.040163,
0.039724,0.039931,0.040126,0.040071,0.040176);

// Mat kernel = (Mat_<double>(5,5) << 4.2005503884146651e-02,	   4.5406518659169053e-02,	   4.5503798031617484e-02	,   4.4631633834065841e-02,	   4.8617907660330625e-02,	  3.7879017155846358e-02,	   4.1096105256568434e-02,	   4.1231878114146941e-02,	   4.0390370016659342e-02	,   4.4218404291115225e-02,	 3.7435979849605658e-02,	   4.0686648854956396e-02,	   4.0921617471341334e-02,	   3.9999246678081725e-02	 ,  4.3827085400541803e-02,	3.5212979020281754e-02,	   3.8261066923625464e-02,	   3.8382260561591851e-02,	   3.7612191304491102e-02	  , 4.1400402964530743e-02,	3.2272031222225044e-02,	   3.5155528137547264e-02,	   3.5201420920913676e-02,	   3.4521039253912679e-02	   ,3.8129364532687490e-02);	


//-------------------------------------------------------
//Mat kernel = (Mat_<double>(10,10) <<  9.7724,	   9.8335,	   9.8752,	   9.9151,	   9.9561,	   9.9890,	   1.0002,	   9.9973,	   9.9717,	   9.9175,	9.7880,	9.8612,	   9.9064,	   9.9474,	   9.9924,	   1.0030,	   1.0048,	   1.0044,	   1.0016,	   9.9478,	9.8007,	   9.8807,	   9.9279,	   9.9709,	 1.0020,	   1.0062,	   1.0081,	   1.0076,	   1.0045,	   9.9682,	9.8144,	   9.8973,	   9.9468,	   9.9922,	   1.0045,	   1.0090,	   1.0107,	   1.0098,	   1.0063,	   9.9803,	9.8288,	   9.9135,	   9.9655,	   1.0014,	   1.0071,	   1.0117,	   1.0130,	   1.0114,	   1.0074,	   9.9860,	9.8411,	   9.9278,	   9.9829,	   1.0034,	   1.0093,	   1.0152,	   1.0145,	   1.0121,	   1.0075,	   9.9837,9.8497,	   9.9377,	   9.9961,	   1.0049,	  1.0106,	   1.0147,	   1.0147,	   1.0115,	   1.0063,	   9.9702,	9.8546,	   9.9422,	   1.0002,	   1.0055,	   1.0107,	   1.0141,	   1.0134,	  1.0095,	   1.0038,	   9.9474,	9.8570,	   9.9398,	   1.0000,	   1.0051,	   1.0097,	   1.0122,	   1.0108,	   1.0062,	   1.0003,	   9.9188,	9.8543,	   9.9260,	   9.9824,	   1.0031,	   1.0071,	   1.0087,	   1.0065,	   1.0014,	   9.9550,	   9.8840);

//kernel=kernel/1000;
//-------------------------------------------------------
//Mat kernel = (Mat_<double>(5,5) << 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
	
// create 2d gaus
	//Mat kernel = kernelX * kernelY.t();
//cout<<kernel.cols<<" "<<kernel.rows;


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

//inputs are the blurry image with noise , the original image power spectra , and standard deviation of the noise introduced
Mat wiener2(Mat final_noise, Mat image_spectrum, int noise_stddev){
	Mat padded = padd_image(final_noise);
	Mat noise = rand_noise(padded, noise_stddev);
	Mat noise_spectrum = get_spectrum(noise);

	Scalar padded_mean = mean(padded);

	Mat planes[2];
	Mat complexI = get_dft(padded);
	split(complexI, planes);	// planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

	Mat factor = (noise_spectrum / image_spectrum); //calculates the signal to noise ratio
//factor=.1;
//cout<<factor;
//cout<<"snr"<<factor<<endl;
//-----------------add H(f)



   Mat mask = createavg(padded.size());
//Mat mask = imread("psf.jpg");
//imwrite("mask.jpg",mask);		//creating the kernel which initally prduced the blurred image----------------- (imread  psf)
shift(mask);// shifting the filter
Mat mplane[] = {Mat_<float>(mask), Mat::zeros(mask.size(), CV_32F)};
Mat kernelcomplex;
    merge(mplane, 2, kernelcomplex); 

dft(kernelcomplex, kernelcomplex);  // computing dft of kernel

split(kernelcomplex, mplane);// splitting the dft of kernel to real and complex 
Mat x= mplane[0].clone();
//cout<<x;

magnitude(mplane[0], mplane[1], mplane[0]);// planes[0] = magnitude
	Mat magI = mplane[0];   
//cout<<magI;
	multiply(magI,magI,magI);        //Computing |H(f)|^2
//<<factor;
factor+=magI;					//adding to signal to noise ratio
//cout<<factor;
magI=magI/factor;   			// calculating 	(|H(f)|^2)/(|H(f)|^2 + S/N)			

//cout<<magI << " "<<x;
 magI=magI/x;				//Dividing by the real value part of dft of kernel thus effectively multiplying by (1/H(f))
factor=magI;	
//cout<<factor;			
//cout<<magI;

//-------------------add H(f)



	multiply(planes[0],factor,planes[0]);
	multiply(planes[1],factor,planes[1]);


	merge(planes, 2, complexI);
//Mat res = updateResult(complexI);
//imshow("res", res);
	idft(complexI, complexI);
	split(complexI, planes);
//	normalize(planes[0], planes[0], 0, 128, CV_MINMAX );
	Scalar enhanced_mean = mean(planes[0]);
	double norm_factor =  padded_mean.val[0] / enhanced_mean.val[0];
	multiply(planes[0],norm_factor, planes[0]);
	Mat normalized;
	planes[0].convertTo(normalized, CV_8UC1);
	return normalized;
}

Mat padd_image(Mat I){
	Mat padded;
	int m = getOptimalDFTSize( I.rows );
	int n = getOptimalDFTSize( I.cols ); // on the border add zero pixels
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
	return padded;
}

Mat get_spectrum(Mat I){
	Mat complexI = get_dft(I);
	Mat planes[2];
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];
	multiply(magI,magI,magI);
	return magI;
}

Mat get_dft(Mat I){
	Mat image;
	I.convertTo(image, CV_32F);
	Mat planes[] = {Mat_<float>(image), Mat::zeros(image.size(), CV_32F)};
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	return complexI;
}

Mat with_noise(Mat image, int stddev){
	Mat noise(image.rows, image.cols, CV_8U);
	rand_noise(image, stddev).convertTo(noise, CV_8U);
	Mat noisy = image.clone();
	noisy += noise;
	return noisy;
}

Mat rand_noise(Mat I, int stddev){
	Mat noise = Mat::zeros(I.rows, I.cols, CV_32F);
	randn(noise,Scalar::all(0), Scalar::all(stddev));
	return noise;
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
