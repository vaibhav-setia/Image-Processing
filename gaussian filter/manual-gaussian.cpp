#include <iostream>
#include <cmath>
#include <iomanip>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
using namespace cv;
using namespace std;
 
double GKernel[5][5];
// Function to create Gaussian filter
void FilterCreation(double GKernel[][5])
{
    // intialising standard deviation to 1.0
    double sigma = 6.0;
    double r, s = 6.0 * sigma * sigma;
 
    // sum is for normalization
    double sum = 0.0;
 
    // generating 5x5 kernel
    for (int x = -2; x <= 2; x++)
    {
        for(int y = -2; y <= 2; y++)
        {
            r = sqrt(x*x + y*y);
            GKernel[x + 2][y + 2] =
                     (exp(-(r*r)/s))/(M_PI * s);
            sum += GKernel[x + 2][y + 2];
        }
    }
 
    // normalising the Kernel
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            GKernel[i][j] /= sum;
}

void masking(Mat image){

Mat temImage= image.clone();
for (int i = 1; i < image.rows-1; i++)
{
    for (int j = 1; j < image.cols-1; j++)
    {   
        for(int k=0;k<3;k++)
        {
	    
int pixel1 = image.at<Vec3b>(i-2,j-2)[k]* GKernel[0][0];
int pixel2 = image.at<Vec3b>(i-1,j-2)[k]* GKernel[1][0];
int pixel3 = image.at<Vec3b>(i,j-2)[k]* GKernel[2][0];
int pixel4 = image.at<Vec3b>(i+1,j-2)[k]* GKernel[3][0];
int pixel5 = image.at<Vec3b>(i+2,j-2)[k]* GKernel[4][0];

int pixel6 = image.at<Vec3b>(i-2,j-1)[k]* GKernel[0][1];
int pixel7 = image.at<Vec3b>(i-1,j-1)[k]* GKernel[1][1];
int pixel8 = image.at<Vec3b>(i,j-1)[k]* GKernel[2][1];
int pixel9 = image.at<Vec3b>(i+1,j-1)[k]* GKernel[3][1];
int pixel10 = image.at<Vec3b>(i+2,j-1)[k]* GKernel[4][1];

int pixel11 = image.at<Vec3b>(i-2,j)[k]* GKernel[0][2];
int pixel12 = image.at<Vec3b>(i-1,j)[k]* GKernel[1][2];
int pixel13 = image.at<Vec3b>(i,j)[k]*  GKernel[2][2];
int pixel14 = image.at<Vec3b>(i+1,j)[k]*  GKernel[3][2];
int pixel15 = image.at<Vec3b>(i+2,j)[k]*  GKernel[4][2];

int pixel16 = image.at<Vec3b>(i-2,j+1)[k]* GKernel[0][3];
int pixel17 = image.at<Vec3b>(i-1,j+1)[k]* GKernel[1][3];
int pixel18 = image.at<Vec3b>(i,j+1)[k]* GKernel[2][3];
int pixel19 = image.at<Vec3b>(i+1,j+1)[k]* GKernel[3][3];
int pixel20 = image.at<Vec3b>(i+2,j+1)[k]* GKernel[4][3];

int pixel21 = image.at<Vec3b>(i-2,j+2)[k]* GKernel[0][4];
int pixel22 = image.at<Vec3b>(i-1,j+2)[k]* GKernel[1][4];        
int pixel23 = image.at<Vec3b>(i,j+2)[k]* GKernel[2][4];
int pixel24 = image.at<Vec3b>(i+1,j+2)[k]*  GKernel[3][4];
int pixel25 = image.at<Vec3b>(i+2,j+2)[k]*  GKernel[4][4];       



     int sum = pixel1 + pixel2 + pixel3 + pixel4 + pixel5 + pixel6 + pixel7 + pixel8 + pixel9 + pixel10 + pixel11 + pixel12 + pixel13 + pixel14 + pixel15 + pixel16 + pixel17 + pixel18 + pixel19 + pixel20 + pixel21 + pixel22 + pixel23 + pixel24 + pixel25;
            if(sum < 0)
            {
                sum = 0;
            }

            if(sum > 255)
                sum = 255;

            temImage.at<Vec3b>(i-1,j-1)[k]= sum;


        }
    }

}
imshow( "Display", temImage );
imwrite("output1.jpg",temImage);

}
 
// Driver program to test above function
int main()
{
    
    FilterCreation(GKernel);
Mat input_image = imread("sample.jpeg" , CV_LOAD_IMAGE_COLOR);

masking(input_image);
waitKey(0);
 
}
