#include "opencv.hpp"
#include "objdetect/objdetect.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#define IMGTHRESHOLD 50

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndSave( Mat frame );

void sharpen(
	cv::Mat &input, 
	int size,
	cv::Mat &out);

/** Global variables */
String logo_cascade_name = "darts_o/original/dartcascade.xml";// haarcascade_frontalface_default.xml";

CascadeClassifier logo_cascade;

string window_name = "Capture - Face detection";

/** @function main */
int main( int argc, const char** argv )
{
	CvCapture* capture;
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	//-- 1. Load the cascades
	//logo_cascade.load(logo_cascade_name);
	if( !logo_cascade.load( logo_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	detectAndSave( frame );

	return 0;
}

void sharpen(cv::Mat &input, int size, cv::Mat &out)
{
	cv::Mat blurredOutput ;
	// intialise the output using the input
	blurredOutput.create(input.size(), CV_64F) ; //input.type());

	// create the Gaussian kernel in 1D 
	//cv::Mat kX = cv::getGaussianKernel(size, -1);
	//cv::Mat kY = cv::getGaussianKernel(size, -1);
	
	// make it 2D multiply one by the transpose of the other
	//cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	//was CV_64FC1
	cv::Mat kernel = cv::Mat(size, size, CV_64F, cv::Scalar::all(-1));
	kernel.at<double>(size/2, size/2) = size*size;

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{	
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + kernelRadiusX + m;
					int imagey = j + kernelRadiusY + n;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					// int - int - uchar
					double imageval = ( double ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;							
				}
			}
			//std::cout << "Sum: "<< sum << " sum uchar " << (uchar)sum << " sum%mod " << (char)sum%255 << std::endl;
			// set the output value as the sum of the convolution

			// if(sum > 255)
			// {
			// 	sum = 255;
			// }
			// else if (sum < 0)
			// {
			// 	sum = 0;
			// }

			blurredOutput.at<double>(i, j) = (double)sum;
		}
	}

	cv::Mat temp8b ;
	cv::normalize(blurredOutput, temp8b, 0, 255, cv::NORM_MINMAX);
	temp8b.convertTo(out, CV_8U);

}


/** @function detectAndSave */
void detectAndSave( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// medianBlur(frame_gray, frame_gray, 3) ;
	imshow("gray",frame_gray);
	waitKey();

	GaussianBlur(frame_gray, frame_gray, Size(3, 3), 1) ; //0.7 was OK
	imshow("gaus blurred",frame_gray);
	waitKey();

	sharpen(frame_gray, 3, frame_gray);
	imshow("sharpen",frame_gray);
	waitKey();

	cv::normalize(frame_gray, frame_gray, 0, 122, cv::NORM_MINMAX);


	// threshold to produce solid black(remove shades)
	for(int i = 0; i < frame_gray.rows; ++i)
	{
		for (int j = 0; j < frame_gray.cols; ++j)
		{
			uchar tr = frame_gray.at<uchar>(i, j);
			// if(tr < 35)
			// {
			// 	frame_gray.at<uchar>(i, j) = 15 ;//0
			// }
			// else if(tr<70)
			// {
			// 	frame_gray.at<uchar>(i, j) = 45;//40
			// }
			// else if (tr<105)
			// {
			// 	frame_gray.at<uchar>(i, j) = 75;//80
			// }
			// else if (tr<140)
			// {
			// 	frame_gray.at<uchar>(i, j) = 105;//120
			// }
///////////////////////////////////////////////////////////////////////////////
			if(tr < 20)
			{
				frame_gray.at<uchar>(i, j) = 10 ;//0
			}
			else if(tr<40)
			{
				frame_gray.at<uchar>(i, j) = 40;//40
			}
			else if (tr<60)
			{
				frame_gray.at<uchar>(i, j) = 80;//80
			}
			else if (tr<80)
			{
				frame_gray.at<uchar>(i, j) = 120;//120
			}
			else if (tr<100)
			{
				frame_gray.at<uchar>(i, j) = 160; //180
			}
			else if (tr<120)
			{
				frame_gray.at<uchar>(i, j) = 200; //180
			}
			else if (tr<140)
			{
				frame_gray.at<uchar>(i, j) = 240; //180
			}
///////////////////////////////////////////////////////////////////////////////
			//BEST EVA
			// if(tr < 50)
			// {
			// 	frame_gray.at<uchar>(i, j) = 10 ;//0
			// }
			// else if(tr<100)
			// {
			// 	frame_gray.at<uchar>(i, j) = 50;//40
			// }
			// else if (tr<150)
			// {
			// 	frame_gray.at<uchar>(i, j) = 90;//80
			// }
			// else if (tr<200)
			// {
			// 	frame_gray.at<uchar>(i, j) = 130;//120
			// }
			// else if (tr<255)
			// {
			// 	frame_gray.at<uchar>(i, j) = 190; //180
			// }
////////////////////////////////////////////////////////////////////////////////


	// 		if (tr < 25)
	// 		{
	// 			frame_gray.at<uchar>(i, j) = 10 ; //10 ;
	// 		}
	// 		else if (tr < 50)
	// 		{
	// 			frame_gray.at<uchar>(i, j) = 35 ; //30 ;
	// 		}
	// 		else if (tr < 75)
	// 		{
	// 			frame_gray.at<uchar>(i, j) = 60 ; //50 ;
	// 		}
	// 		else if (tr < 100)
	// 		{
	// 			frame_gray.at<uchar>(i, j) =  85; //70 ;
	// 		}
	// 		else if (tr < 125)
	// 		{
	// 			frame_gray.at<uchar>(i, j) =  110; //90 ;
	// 		}
	// 		else if (tr < 150)
	// 		{
	// 			frame_gray.at<uchar>(i, j) =  135; //110 ;
	// 		}
	// 		else if (tr < 175)
	// 		{
	// 			frame_gray.at<uchar>(i, j) =  160; //130 ;
	// 		}
	// 		else if (tr < 200)
	// 		{
	// 			frame_gray.at<uchar>(i, j) =  185; //150 ;
	// 		}
	// 		else if (tr < 225)
	// 		{
	// 			frame_gray.at<uchar>(i, j) =  210; //160 ;
	// 		}
	// 		else if (tr < 255)
	// 		{
	// 			frame_gray.at<uchar>(i, j) =  240; //180 ;
	// 		}
			// else if (tr < 210)
			// {
			// 	frame_gray.at<uchar>(i, j) =  195; //200 ;
			// }
			// else if (tr < 230)
			// {
			// 	frame_gray.at<uchar>(i, j) =  215; //220 ;
			// }
			// else if (tr < 250)
			// {
			// 	frame_gray.at<uchar>(i, j) =  235; //240 ;
			// }
			// else if (tr < 255)
			// {
			// 	frame_gray.at<uchar>(i, j) =  250; //255 ;
			// }
			// else
			// {
			// 	frame_gray.at<uchar>(i, j) = 0 ;
			// }
		}
	}
	imshow("prev",frame_gray);
	waitKey();

	// Blur the image to smooth the noise
	Mat blurred ;
	GaussianBlur(frame_gray, blurred, Size(3, 3), 1.2) ;
	// medianBlur(blurred, blurred, 3) ;

	//-- Detect faces
	logo_cascade.detectMultiScale( blurred, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	std::cout << faces.size() << std::endl;

	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	//-- Save what you got
	imwrite( "output.jpg", frame );

}
