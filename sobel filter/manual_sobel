#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

char mask [3][3]=  {{-1,-2,-1},{0,0,0},{1,2,1}}; 

void masking(Mat image){

Mat temImage= image.clone();
for (int i = 1; i < image.rows-1; i++)
{
    for (int j = 1; j < image.cols-1; j++)
    {   
       
            int pixel1 = image.at<uchar>(i-1,j-1)* -1;
            int pixel2 = image.at<uchar>(i,j-1)* -2;
            int pixel3 = image.at<uchar>(i+1,j-1)* -1;

            int pixel4 = image.at<uchar>(i-1,j)* 0;
            int pixel5 = image.at<uchar>(i,j)* 0;
            int pixel6 = image.at<uchar>(i+1,j)* 0;

            int pixel7 = image.at<uchar>(i-1,j+1)* 1;
            int pixel8 = image.at<uchar>(i,j+1)* 2;
            int pixel9 = image.at<uchar>(i+1,j+1)* 1;

	    int pixel11 = image.at<uchar>(i-1,j-1)* -1;
            int pixel21 = image.at<uchar>(i,j-1)* 0;
            int pixel31 = image.at<uchar>(i+1,j-1)* 1;

            int pixel41 = image.at<uchar>(i-1,j)* -2;
            int pixel51 = image.at<uchar>(i,j)* 0;
            int pixel61 = image.at<uchar>(i+1,j)* 2;

            int pixel71 = image.at<uchar>(i-1,j+1)* -1;
            int pixel81 = image.at<uchar>(i,j+1)* 0;
            int pixel91 = image.at<uchar>(i+1,j+1)* 1;


            int sum = pixel1 + pixel2 + pixel3 + pixel4 + pixel5 + pixel6 + pixel7 + pixel8 + pixel9;
  int sum1 = pixel11 + pixel21 + pixel31 + pixel41 + pixel51 + pixel61 + pixel71 + pixel81 + pixel91;


sum+=sum1;
           if(sum < 0)
            {
                sum = 0;
            }

            if(sum > 255)
                sum = 255;

            temImage.at<uchar>(i,j)= sum;


        
    }

}
imshow( "Display", temImage );

}
int main( int argc, char** argv ){
Mat src_gray;
Mat input_image = imread("sample.jpeg" , CV_LOAD_IMAGE_COLOR);
cvtColor( input_image, src_gray, CV_BGR2GRAY );
masking(src_gray);
waitKey(0);
return 0;

}
