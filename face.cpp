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

/** @function detectAndSave */
void detectAndSave( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	medianBlur(frame_gray, frame_gray, 3) ;

	// threshold to produce solid black(remove shades)
	for(int i = 0; i < frame_gray.rows; ++i)
	{
		for (int j = 0; j < frame_gray.cols; ++j)
		{
			uchar tr = frame_gray.at<uchar>(i, j);
			if (tr < 20)
			{
				frame_gray.at<uchar>(i, j) = 5 ; //10 ;
			}
			else if (tr < 40)
			{
				frame_gray.at<uchar>(i, j) = 25 ; //30 ;
			}
			else if (tr < 60)
			{
				frame_gray.at<uchar>(i, j) = 45 ; //50 ;
			}
			else if (tr < 80)
			{
				frame_gray.at<uchar>(i, j) =  65; //70 ;
			}
			else if (tr < 100)
			{
				frame_gray.at<uchar>(i, j) =  85; //90 ;
			}
			else if (tr < 120)
			{
				frame_gray.at<uchar>(i, j) =  105; //110 ;
			}
			else if (tr < 140)
			{
				frame_gray.at<uchar>(i, j) =  125; //130 ;
			}
			else if (tr < 160)
			{
				frame_gray.at<uchar>(i, j) =  145; //150 ;
			}
			else if (tr < 170)
			{
				frame_gray.at<uchar>(i, j) =  155; //160 ;
			}
			else if (tr < 190)
			{
				frame_gray.at<uchar>(i, j) =  175; //180 ;
			}
			else if (tr < 210)
			{
				frame_gray.at<uchar>(i, j) =  195; //200 ;
			}
			else if (tr < 230)
			{
				frame_gray.at<uchar>(i, j) =  215; //220 ;
			}
			else if (tr < 250)
			{
				frame_gray.at<uchar>(i, j) =  235; //240 ;
			}
			else if (tr < 255)
			{
				frame_gray.at<uchar>(i, j) =  250; //255 ;
			}
			// else
			// {
			// 	frame_gray.at<uchar>(i, j) = 0 ;
			// }
		}
	}

	// Blur the image to smooth the noise
	Mat blurred ;
	GaussianBlur(frame_gray, blurred, Size(3, 3), 1.2) ;
	medianBlur(blurred, blurred, 3) ;

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
