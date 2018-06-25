#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

char mask [3][3]=  {{1/9,1/9,1/9},{1/9,1/9,1/9},{1/9,1/9,1/9}}; 

void masking(Mat image){

Mat temImage= image.clone();
for (int i = 1; i < image.rows-1; i++)
{
    for (int j = 1; j < image.cols-1; j++)
    {   
        for(int k=0;k<3;k++)
        {
            int pixel1 = image.at<Vec3b>(i-1,j-1)[k]* 1/9;
            int pixel2 = image.at<Vec3b>(i,j-1)[k]* 1/9;
            int pixel3 = image.at<Vec3b>(i+1,j-1)[k]* 1/9;

            int pixel4 = image.at<Vec3b>(i-1,j)[k]* 1/9;
            int pixel5 = image.at<Vec3b>(i,j)[k]* 1/9;
            int pixel6 = image.at<Vec3b>(i+1,j-1)[k]* 1/9;

            int pixel7 = image.at<Vec3b>(i-1,j+1)[k]* 1/9;
            int pixel8 = image.at<Vec3b>(i,j+1)[k]* 1/9;
            int pixel9 = image.at<Vec3b>(i+1,j+1)[k]* 1/9;

            int sum = pixel1 + pixel2 + pixel3 + pixel4 + pixel5 + pixel6 + pixel7 + pixel8 + pixel9;
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
//printf("conter = %d",counter);
imshow( "Display", temImage );
imwrite("output1.jpg",temImage);

}
int main( int argc, char** argv ){
Mat src_gray;
Mat input_image = imread("sample.jpeg" , CV_LOAD_IMAGE_COLOR);
//cvtColor( input_image, src_gray, CV_BGR2GRAY );
masking(input_image);
waitKey(0);
return 0;

}
