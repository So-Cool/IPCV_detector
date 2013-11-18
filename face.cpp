#include "opencv.hpp"
#include "objdetect/objdetect.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"
#include "filters.hpp"

#include <iostream>
#include <stdio.h>
#include <math.h>       /* pow */



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
	std::vector<Rect> faces, faces1, faces2;
	Mat frame_gray;
	double average = 0;
	double  count = 0;

	// for temporary storage of Mat construct
	cv::Mat temp8Bit;

	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	Mat original = frame_gray.clone();

	// delete dark colors (preparation for histogram stretching)
	Mat bright(frame_gray.rows, frame_gray.cols, CV_8U, cv::Scalar::all(0));
	Mat dark(frame_gray.rows, frame_gray.cols, CV_8U, cv::Scalar::all(0));
	for (int i = 0; i < frame_gray.rows; ++i)
	{
		for (int j = 0; j < frame_gray.cols; ++j)
		{
			average += frame_gray.at<uchar>(i,j);
			++count;
			if (frame_gray.at<uchar>(i,j) < 125)//220 pretty good
			{
				bright.at<uchar>(i,j) = 125  ;
			}
			else
			{
				bright.at<uchar>(i,j) = frame_gray.at<uchar>(i,j) ;
			}


			if (frame_gray.at<uchar>(i,j) > 125)//220 pretty good
			{
				dark.at<uchar>(i,j) = 125  ;
			}
			else
			{
				dark.at<uchar>(i,j) = frame_gray.at<uchar>(i,j) ;
			}
		}
	}
	cout<< "average gray level: " << average/count << endl;

	// Darken image
	// Stretch the histogram from 125-255 to 0-255
	// As effect we get overall darken image
	Mat darken;
	cv::normalize(bright, darken, 0, 255, cv::NORM_MINMAX);
	imshow("Dark", darken);
	waitKey();

	Mat brighten;
	cv::normalize(dark, brighten, 0, 255, cv::NORM_MINMAX);
	imshow("Bright", brighten);
	waitKey();

	// detect circles and print them
	cv::Mat xDeriveCRC, yDeriveCRC, gradCRC, arcCRC, outCRC;
	sobel(darken, xDeriveCRC, yDeriveCRC, gradCRC, arcCRC); // original
	std::vector<std::vector<std::vector<int> > > roundShapes = detectCircles(gradCRC, arcCRC, outCRC);

	// threshold Hough space
	for (int i = 0; i < outCRC.rows; ++i)
	{
		for (int j = 0; j < outCRC.cols; ++j)
		{
			if (outCRC.at<uchar>(i,j) < CIRCLETHRESHOLD)//220 pretty good
			{
				outCRC.at<uchar>(i,j) = 0  ;
			}
		}
	}

	// Apply Mexican Hat filter
	mexHat(outCRC, temp8Bit);
	imshow("CRC & MexHat", temp8Bit);
	waitKey();


	// Blur the image to smooth the noise
	Mat medianB, gaussianB5, gaussianB3, gaussianB1;
	// medianBlur(frame_gray, frame_gray, 3) ;
	GaussianBlur(darken, gaussianB5, Size(5, 5), 0.7) ; // 1.2 //0.7 was OK
	GaussianBlur(original, gaussianB3, Size(5, 5), 1.2) ; // 1.2 //0.7 was OK // for wall only
	sharpen(gaussianB5, 3, gaussianB5);
	sharpen(gaussianB3, 3, gaussianB3);
	gaussianB1 = original.clone();

	// imshow("gausblur", gaussianB5);
	// waitKey();


	//-- Detect faces
	// logo_cascade.detectMultiScale( gaussianB5, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) ); //blurred
	// logo_cascade.detectMultiScale( gaussianB3, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) ); //blurred
	logo_cascade.detectMultiScale( gaussianB1, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) ); //blurred
	std::cout << faces.size() << std::endl;

	for( int i = 0; i < faces.size(); i++ )
	{
		// Mat tmp;
		// extractRegion(original, tmp, faces[i].x, faces[i].y, faces[i].width);
		// if(!checkHomogenity(tmp))
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	//-- Save what you got
	imshow("output",frame);
	waitKey();
	imwrite( "output.jpg", frame );





////////////////////////////////////////////////////////////////////////////////




	// // Adoptive thresholding
	// Mat thresholded(frame_gray.rows, frame_gray.cols, CV_8U, cv::Scalar::all(0));
	// nLvlTrsh(frame_gray, thresholded);
	// imshow("Thrsh", thresholded);
	// waitKey();


	// threshold to produce solid black(remove shades)
	// for(int i = 0; i < frame_gray.rows; ++i)
	// {
	// 	for (int j = 0; j < frame_gray.cols; ++j)
	// 	{
	// 		uchar tr = frame_gray.at<uchar>(i, j);
	// 		if(tr < 75)
	// 		{
	// 			frame_gray.at<uchar>(i, j) = 0 ;
	// 		}

	// 	}
	// }
	// if(SHOW) imshow("prev",frame_gray);
	// if(SHOW) waitKey();


	//detect lines
	// cv::Mat xDeriv, yDeriv, grad_ory, grad_trs, arc_ory, arc_trs, output, out2, outcirc ;//frame_gray
	// sobel(darken, xDeriv, yDeriv, grad_ory, arc_ory);//original
	// sobel(sharpened, xDeriv, yDeriv, grad_trs, arc_trs);//original
	// detectCircles(grad_ory, arc_ory, outcirc);
	// detectLines(grad_trs, arc_trs, output);
	//detect lines only in circle with adaptive threshold
	//in selectedby circles regionns look for line hough spectrum similar to the one of dartboard.bmp
	//najlepiej zrob thersholiding and XOR
	// extractRegion(original, out2, 50, 50, 20);
	// if (checkHomogenity(original)) cout<<"homogeneous" << endl;
	// else cout<<"IN-homogeneous"<<endl;


////////////////////////////////////////////////////////////////////////////////
	// cv::Mat temp8Bit;
	// cv::normalize(outcirc, temp8Bit, 0, 255, cv::NORM_MINMAX);
	// temp8Bit.convertTo(outcirc, CV_8U);
	// for (int i = 0; i < outcirc.rows; ++i)
	// {
	// 	for (int j = 0; j < outcirc.cols; ++j)
	// 	{
	// 		if (outcirc.at<uchar>(i,j) < CIRCLETHRESHOLD)//220 pretty good
	// 		{
	// 			outcirc.at<uchar>(i,j) = 0  ;
	// 		}
	// 	}
	// }
	// mexHat(outcirc, output);
	// imshow("mex", output);
	// waitKey();
////////////////////////////////////////////////////////////////////////////////

}
