#include "opencv.hpp"
#include "objdetect/objdetect.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <math.h>       /* pow */

#define IMGTHRESHOLD 50
#define HOUGHDETECTTRESHOLD 70

//line
//5 degrees is 0,087
#define DTH 9 //delta angle * 100
#define LINETHRESHOLD 235

//circle
#define RMIN 5
#define RMAX 75
#define CIRCLETHRESHOLD 220

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

			blurredOutput.at<double>(i, j) = (double)sum;
		}
	}

	cv::Mat temp8b ;
	cv::normalize(blurredOutput, temp8b, 0, 255, cv::NORM_MINMAX);
	temp8b.convertTo(out, CV_8U);

}

//line detect
void sobel(const cv::Mat& image, cv::Mat& xDeriv, cv::Mat& yDeriv, cv::Mat& grad, cv::Mat& arc)
{
	double minVal, maxVal;
	cv::Sobel(image, xDeriv, CV_64F, 1, 0);
	cv::Sobel(image, yDeriv, CV_64F, 0, 1);
	cv::Mat xPow, yPow;
	cv::pow(xDeriv, 2, xPow);
	cv::pow(yDeriv, 2, yPow);
	cv::Mat temp = xPow + yPow;
	cv::Mat tempFloat;
	temp.convertTo(tempFloat, CV_64F);
	cv::Mat gradNorm;
	cv::sqrt(tempFloat, grad);
	cv::Mat divided, arcNorm;
	arc = ( cv::Mat(xDeriv.rows, xDeriv.cols, CV_64F) ).clone() ;
	cv::divide(yDeriv, xDeriv, divided);
	cout << "sobel doing" << endl;
	for(int i = 0; i < divided.rows; i++)
	{
		for(int j =0; j < divided.cols; j++)
		{	
			arc.at<double>(i, j) = (double)atan2(yDeriv.at<double>(i,j), xDeriv.at<double>(i,j)) ;//* 180 / PI;
		}
	}
	cout << "sobel done" << endl;
}



void detectLines(const cv::Mat& grad, const cv::Mat& arc, cv::Mat& out)
{
	// cv::Mat lineHoughSpace = cv::Mat(grad.rows, grad.cols, CV_64F, cv::Scalar::all(0));

	// theta ranges from 0-2PI = 0 - 628
	// tho is dynamically adjusted from 0 to max_val based on diagonal
	int diagonalSize = round(sqrt((double)std::pow((double)(grad.rows), 2) + (double)std::pow((double)(grad.cols),2)));
	// cout << diagonalSize << "  " << grad.rows << "  " << grad.cols << endl ;
	diagonalSize *= 2; 

	std::vector<std::vector<int> > houghSpace (diagonalSize, std::vector<int>(628, 0) ) ;
	cv::Mat lineHoughSpace = cv::Mat(diagonalSize, 628, CV_64F, cv::Scalar::all(0)); //628

	int ujemne = 0;
	int dodatnie = 0;
	double rem1 = 0;
	double rem2 = 0;

	for(int i = 0; i < grad.rows; ++i)
	{
		for (int j = 0; j < grad.cols; ++j)
		{
			if (grad.at<double>(i, j) > HOUGHDETECTTRESHOLD)
			{

				//LINE DETECTION
				// for (int th = round(arc.at<double>(i, j)-DTH); th < round(arc.at<double>(i, j)+DTH); ++th)
				int trows = lineHoughSpace.rows ;
				int tcols = lineHoughSpace.cols ;
				for (int th = (arc.at<double>(i,j)*100)-DTH; th < (arc.at<double>(i,j)*100)+DTH; ++th)
				{
					if (th<0) continue;
					// cout << double(th)/100 << endl;
					// double rho = i * cos(double(th)/100)+ j*sin(double(th)/100) ; //first
					double rho = j * cos(double(th)/100)+ i*sin(double(th)/100) ; //second
					// cout << rho+diagonalSize/2 << endl;

					//invrease haff pace
					// if(round(double(rho/10))<trows && round(double(rho/10))>=0 && round(double(th)/10)<tcols && round(double(th/10))>0)
					// {
					// 	lineHoughSpace.at<double>(round(double(rho/10)), round(double(th/10)) ) += 1 ;
					// }

					if (rho+diagonalSize/2 <0)
					{
						ujemne++;
					} else {
						dodatnie++;
					}
					if (rho<rem1) rem1 = rho;
					if (rho>rem2) rem2 = rho;

					// create hough space
					// if(round(double(rho/10))<trows && round(double(rho/10))>=0 && round(double(th)/10)<tcols && round(double(th/10))>0)
					// {
						// cout << "first" << round(th) << " " << round(rho+diagonalSize/2) << endl;
						houghSpace[round(rho+diagonalSize/2)][round(th) ] += 1 ;
						// cout << "second" << round(th) << " " << round(rho+diagonalSize/2) << endl;
						lineHoughSpace.at<double>(round(rho+diagonalSize/2), round(th)) += 1 ;
					// }

				}
			}
		}
	}

	// cout << "ujemne: " << ujemne << " " << dodatnie << endl;
	// cout << rem1 << "  " << rem2 << endl;

	cout << "tu" << endl;

	//print haff space fol LINE
	//take logs
	for (int i = 0; i < lineHoughSpace.rows; ++i)
	{
		for (int j = 0; j < lineHoughSpace.cols; ++j)
		{
			if (lineHoughSpace.at<double>(i,j) != 0)
			{
				lineHoughSpace.at<double>(i,j) = log( lineHoughSpace.at<double>(i,j) ) ;
			}
		}
	}
	//scale
	cv::Mat temp8Bit;
	cv::normalize(lineHoughSpace, temp8Bit, 0, 255, cv::NORM_MINMAX);
	temp8Bit.convertTo(lineHoughSpace, CV_8U);
	// for (int i = 0; i < lineHoughSpace.rows; ++i)
	// {
	// 	for (int j = 0; j < lineHoughSpace.cols; ++j)
	// 	{
	// 		if (lineHoughSpace.at<uchar>(i,j) < LINETHRESHOLD)//220 pretty good
	// 		{
	// 			lineHoughSpace.at<uchar>(i,j) = 0  ;
	// 		}
	// 	}
	// }
	//convert
	cv::imshow("Hough space", lineHoughSpace) ;
	waitKey();
}



void detectCircles(const cv::Mat& grad, const cv::Mat& arc, cv::Mat& out)
{
	// cv::vector<cv::Vec3d> circles ; //x,y,r
	// std::vector<std::vector<std::vector<int> > > houghSpace (HOUGHY, std::vector<std::vector<int> > (HOUGHX, std::vector<int>(RMAX-RMIN, 0) ) ) ;
	cv::Mat circleHoughSpace = cv::Mat(grad.rows, grad.cols, CV_64F, cv::Scalar::all(0));
	// threshold the gradient image after normalization
	// cv::Mat gradNorm(grad.rows, grad.cols, CV_64F) ;

	for(int i = 0; i < grad.rows; ++i)
	{
		for (int j = 0; j < grad.cols; ++j)
		{
			if (grad.at<double>(i, j) > HOUGHDETECTTRESHOLD)
			{

				// CIRCLE DETECTION
				for (int r = RMIN; r < RMAX; ++r)
				{
					//shifted by RMAX to make scaling easier task
					double x1 = j+r*cos(arc.at<double>(i,j)) ;
					double x2 = j-r*cos(arc.at<double>(i,j)) ;
					double y1 = i+r*sin(arc.at<double>(i,j)) ;
					double y2 = i-r*sin(arc.at<double>(i,j)) ;

					int trows = circleHoughSpace.rows ;
					int tcols = circleHoughSpace.cols ;

					if ( round(y1)<trows && round(y1)>0 && round(x1)>0 && round(x1)<tcols )
					{
						circleHoughSpace.at<double>(round(y1), round(x1) ) += 1 ;
						// houghSpace[y1*HOUGHY/trows][x1*HOUGHX/tcols][r-RMIN] += 1 ;
					}
					if ( round(y1)<trows && round(y1)>0 && round(x2)>0 && round(x2)<tcols )
					{
						circleHoughSpace.at<double>( round(y1), round(x2)  ) += 1 ;
						// houghSpace[y1*HOUGHY/trows][x2*HOUGHX/tcols][r-RMIN] += 1 ;
					}
					if ( round(y2)<trows && round(y2)>0 && round(x1)>0 && round(x1)<tcols )
					{
						circleHoughSpace.at<double>(  round(y2), round(x1)  ) += 1 ;
						// houghSpace[y2*HOUGHY/trows][x1*HOUGHX/tcols][r-RMIN] += 1 ;
					}
					if ( round(y2)<trows && round(y2)>0 && round(x2)>0 && round(x2)<tcols )
					{
						circleHoughSpace.at<double>(  round(y2), round(x2)  ) += 1 ;
						// houghSpace[y2*HOUGHY/trows][x2*HOUGHX/tcols][r-RMIN] += 1 ;

					}
				}
			}
		}
	}


	//print haff space fol CIRCLE
	//take logs
	cv::Mat temp8Bit;
	for (int i = 0; i < circleHoughSpace.rows; ++i)
	{
		for (int j = 0; j < circleHoughSpace.cols; ++j)
		{
			if (circleHoughSpace.at<double>(i,j) != 0)
			{
				circleHoughSpace.at<double>(i,j) = log( circleHoughSpace.at<double>(i,j) ) ;
			}
		}
	}
	//scale
	cv::normalize(circleHoughSpace, temp8Bit, 0, 255, cv::NORM_MINMAX);
	temp8Bit.convertTo(circleHoughSpace, CV_8U);
	for (int i = 0; i < circleHoughSpace.rows; ++i)
	{
		for (int j = 0; j < circleHoughSpace.cols; ++j)
		{
			if (circleHoughSpace.at<uchar>(i,j) < CIRCLETHRESHOLD)//220 pretty good
			{
				circleHoughSpace.at<uchar>(i,j) = 0  ;
			}
		}
	}
	//convert
	cv::imshow("Hough space", circleHoughSpace) ;
	waitKey();

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

	Mat ory = frame_gray.clone();

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
			if(tr < 75)
			{
				frame_gray.at<uchar>(i, j) = 0 ;
			}

		}
	}
	imshow("prev",frame_gray);
	waitKey();


	//detect lines
	cv::Mat xDeriv, yDeriv, grad, arc, output ;//frame_gray
	sobel(ory, xDeriv, yDeriv, grad, arc);
	detectCircles(grad, arc, output);
	detectLines(grad, arc, output);
	//detect lines only in circle with adaptive threshold
	//in selectedby circles regionns look for line hough spectrum similar to the one of dartboard.bmp
	//najlepiej zrob thersholiding and XOR


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
