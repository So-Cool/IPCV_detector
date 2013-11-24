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

	// read teemplate
	// not needed?
	Mat templat = imread("HoughTemplate.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	for (int x = 0; x < templat.rows; ++x)
	{
		for (int y = 0; y < templat.cols; ++y)
		{
			// black if the same
			if (templat.at<uchar>(x,y) < 125)
			{
				templat.at<uchar>(x,y) = 0;
			}
			//white if different
			else
			{
				templat.at<uchar>(x,y) = 255;
			}
		}
	}

	std::vector<Rect> faces, faces1, faces2, facesSmall, brightSquares;
	Mat frame_gray;


	double average = 0;
	double  count = 0;

	// for temporary storage of Mat construct
	cv::Mat temp8Bit;

	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	Mat original = frame_gray.clone();
	int imageRows = original.rows;
	int imageCols = original.cols;

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

	Mat brighten;
	cv::normalize(dark, brighten, 0, 255, cv::NORM_MINMAX);

	// detect circles and print them
	cv::Mat xDeriveCRC, yDeriveCRC, gradCRC, arcCRC;
	sobel(darken, xDeriveCRC, yDeriveCRC, gradCRC, arcCRC); // original


	//do while found one circle or 10 timesLLLllLLLLLLLLLLLLLLLLLLLLLLLLLL

	int iteration = 0;

	vector<int> deleteme;
	std::vector<cv::Point> brightSpots;
	std::vector<int> brightR;
	std::vector<int> brightValue;


	//choose the coordinates of brightest points
	int rowsmax = 0;
	int colsmax = 0;
	int radmax = 0;
	int vmax = 0;
	int tempmax = 0;
	bool notfound = true;

	do
	{
		cv::Mat outCRC;
		std::vector<std::vector<std::vector<int> > > roundShapes = detectCircles(gradCRC, arcCRC, outCRC, iteration);

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
		mexHat(temp8Bit, temp8Bit);

		rowsmax = 0;
		colsmax = 0;
		radmax = 0;
		vmax = 0;
		tempmax = 0;
		notfound = true;

		deleteme.clear();
		brightSpots.clear();
		brightR.clear();
		brightValue.clear();

		int trows = outCRC.rows ;
		int tcols = outCRC.cols ;

		// find circles
		for (int i = 0; i < temp8Bit.rows; ++i)
		{
			for (int j = 0; j < temp8Bit.cols; ++j)
			{
				if (temp8Bit.at<uchar>(i,j) > CIRCLETHRESHOLD  )
				{
					rowsmax = 0;
					colsmax = 0;
					radmax = 0;
					vmax = 0;
					tempmax = 0;

					for (int k = 0; k < RMAX-RMIN; ++k)
					{
						if ( roundShapes[round(double(i)/double(trows)*CIRCLEROWS)][round(double(j)/double(tcols)*CIRCLECOLS)][k] > vmax)
						{
							vmax = roundShapes[round(double(i)/double(trows)*CIRCLEROWS)][round(double(j)/double(tcols)*CIRCLECOLS)][k];
							radmax = k;
						}
					}

					if (radmax > DISCARDRADIUS)
					{

						// if list empty put first element
						if (brightSpots.size() == 0 && radmax*2<imageCols && radmax*2<imageRows
							&& i+radmax<imageRows && i-radmax>0 &&  j+radmax<imageCols && j-radmax>0)
						{
							brightSpots.push_back(cv::Point(i, j));
							brightR.push_back(radmax);
							brightValue.push_back(vmax);
						}
						else
						{
							//if there is a spot close enough don't put it
							for (int l = 0; l < brightSpots.size(); ++l)
							{
								// erase the 6th element
							    notfound = true;
								if  ( abs(brightSpots[l].x - i ) < PROXIMITY &&
										abs( brightSpots[l].y - j ) < PROXIMITY)
								{
									if( roundShapes[round(double(i)/double(trows)*CIRCLEROWS)][round(double(j)/double(tcols)*CIRCLECOLS)][radmax] > roundShapes[round(double(brightSpots[l].x)/double(trows)*CIRCLEROWS)][round(double(brightSpots[l].y)/double(tcols)*CIRCLECOLS)][brightR[l]]
										&& radmax*2<imageCols && radmax*2<imageRows
										&& i+radmax<imageRows && i-radmax>0 &&  j+radmax<imageCols && j-radmax>0 )
									{
										brightSpots.erase(brightSpots.begin()+ l);
										brightR.erase(brightR.begin()+ l);
										brightValue.erase(brightValue.begin()+ l);

										notfound=true;
										break;
									}
									else
									{
										notfound = false;
										break;
									}
								}
							}

							// add new if not fount
							if (notfound && radmax*2<imageCols && radmax*2<imageRows
								&& i+radmax<imageRows && i-radmax>0 &&  j+radmax<imageCols && j-radmax>0)
							{
								brightSpots.push_back(cv::Point(i, j));
								brightR.push_back(radmax);
								brightValue.push_back(vmax);
							}

						}
					}
				}
			}
		}


		// delete redundant circles by inspecting interior
		Mat extCRC, extCRC_xDeriv, extCRC_yDeriv, extCRC_grad, extCRC_arc;
		for (int i = 0; i < brightSpots.size(); ++i)
		{
			extractRegion(original, extCRC, brightSpots[i].y- brightR[i], brightSpots[i].x- brightR[i], brightR[i]*2);//darkened
			sobel(extCRC, extCRC_xDeriv, extCRC_yDeriv, extCRC_grad, extCRC_arc);//original
			detectLines(extCRC_grad, extCRC_arc, extCRC);

			for (int j = 0; j < extCRC.rows; ++j)
			{
				for (int k = 0; k < extCRC.cols; ++k)
				{
					if (extCRC.at<uchar>(j,k) < LINETHRESHOLD-120)//-120
					{
						extCRC.at<uchar>(j,k) = 0;
					}
					else
					{
						extCRC.at<uchar>(j,k) = 255;
					}
				}
			}

			int white = 0;
			for (int x = 0; x < extCRC.rows; ++x)
			{
				for (int y = 0; y < extCRC.cols; ++y)
				{
					// template white & 
					if (255 == templat.at<uchar>(x,y) && extCRC.at<uchar>(x,y) == 255 )
					{
						white++;
					}
				}
			}

			if (white < DELETECIRCLE)
			{
				//not a circle
				deleteme.push_back(i);
			}
		}

		//delete gathered circles
		for (int i = deleteme.size()-1; i >= 0; --i)
		{
			brightSpots.erase(brightSpots.begin()+deleteme[i]);	
		}

		++iteration;

	} while( brightSpots.size() == 0 && iteration < 10 );

    // providing a negative number will create a filled circle
    int thickness = 2;
	// Parameters for circle
    cv::Scalar redColour(255, 0, 0);
    // 8-connected line
    int linetype = 8; 

	// print circles
	for (int i = 0; i < brightSpots.size(); ++i)
	{
		cout << "CIRC: " << brightSpots[i].x << " " << brightSpots[i].y << " " << brightR[i] << endl;
	    int radius = brightR[i];
		cv::Point center( brightSpots[i].y, brightSpots[i].x );
		cv::circle ( frame , center , radius , redColour , thickness , linetype );
	}




	// Blur the image to smooth the noise
	Mat medianB, gaussianB5, gaussianB3, gaussianB1;
	GaussianBlur(darken, gaussianB5, Size(5, 5), 0.7) ; // 1.2 //0.7 was OK
	GaussianBlur(original, gaussianB3, Size(5, 5), 1.2) ; // 1.2 //0.7 was OK // for wall only
	sharpen(gaussianB5, 3, gaussianB5);
	sharpen(gaussianB3, 3, gaussianB3);
	gaussianB1 = original.clone();

	//-- Detect faces
	logo_cascade.detectMultiScale( gaussianB5, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) ); //blurred
	logo_cascade.detectMultiScale( gaussianB3, faces1, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) ); //blurred
	logo_cascade.detectMultiScale( gaussianB1, faces2, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) ); //blurred
	std::cout << faces.size() + faces1.size() + faces2.size() << std::endl;

	faces1.insert( faces1.end(), faces2.begin(), faces2.end() );
	Mat tmp;

	// lines
	cv::Mat line_xDeriv, line_yDeriv, line_grad, line_arc;
	bool breakable = false;

	for( int i = 0; i < faces.size(); i++ )
	{
		breakable = false;
		extractRegion(darken, tmp, faces[i].x, faces[i].y, faces[i].width);
		if(checkHomogenity(tmp))//size matters
			continue;

		for (int j = 0; j < brightSpots.size(); ++j)
		{
			if (brightSpots[j].y > faces[i].x && brightSpots[j].y < faces[i].x+faces[i].width
				&& brightSpots[j].x > faces[i].y && brightSpots[j].x < faces[i].y+faces[i].height )
			{
				breakable = true;
				break;
			}

		}
		if (breakable)
		{
			continue;
		}

		brightSquares.push_back(faces[i]);
	}
	for( int i = 0; i < faces1.size(); i++ )
	{
		breakable = false;
		extractRegion(original, tmp, faces1[i].x, faces1[i].y, faces1[i].width);
		if(checkHomogenity(tmp))//size matters
			continue;

		for (int j = 0; j < brightSpots.size(); ++j)
		{
			if (brightSpots[j].y > faces1[i].x && brightSpots[j].y < faces1[i].x+faces1[i].width
				&& brightSpots[j].x > faces1[i].y && brightSpots[j].x < faces1[i].y+faces1[i].height )
			{
				breakable = true;
				break;
			}
		}
		if (breakable)
		{
			continue;
		}

		brightSquares.push_back(faces1[i]);
	}

	//delete overlapping squares
	//erase vectpr
	deleteme.clear();

	for (int j = 0; j < brightSquares.size(); ++j)
	{
		// if i belongs to other (j) with some margin delete and break
		for (int i = j+1; i < brightSquares.size(); ++i)
		{
			//similar top-left and bottom right
			if ( abs( brightSquares[i].y - brightSquares[j].y ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].x - brightSquares[j].x ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].y + brightSquares[i].height - brightSquares[j].y -brightSquares[j].height ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].x + brightSquares[i].width - brightSquares[j].x -brightSquares[j].width ) < SQUARETHRESHOLD )
			{
				deleteme.push_back(i);
				continue;
			}


			//similar top-left and bottom-right inside(i)
			if ( abs( brightSquares[i].y - brightSquares[j].y ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].x - brightSquares[j].x ) < SQUARETHRESHOLD &&
				brightSquares[i].y + brightSquares[i].height < brightSquares[j].y +brightSquares[j].height + SQUARETHRESHOLD &&
				brightSquares[i].x + brightSquares[i].width < brightSquares[j].x +brightSquares[j].width + SQUARETHRESHOLD )
			{
				deleteme.push_back(i);
				continue;
			}

			//top-left inside and bottom-right similar
			if ( brightSquares[i].y > brightSquares[j].y - SQUARETHRESHOLD &&
				brightSquares[i].x > brightSquares[j].x - SQUARETHRESHOLD &&
				abs( brightSquares[i].y + brightSquares[i].height - brightSquares[j].y -brightSquares[j].height ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].x + brightSquares[i].width - brightSquares[j].x -brightSquares[j].width ) < SQUARETHRESHOLD )
			{
				deleteme.push_back(i);
				continue;
			}

			//top-left inside and bottom-right inside
			if ( brightSquares[i].y > brightSquares[j].y - SQUARETHRESHOLD &&
				brightSquares[i].x > brightSquares[j].x - SQUARETHRESHOLD &&
				brightSquares[i].y + brightSquares[i].height < brightSquares[j].y +brightSquares[j].height + SQUARETHRESHOLD &&
				brightSquares[i].x + brightSquares[i].width < brightSquares[j].x +brightSquares[j].width + SQUARETHRESHOLD )
			{
				deleteme.push_back(i);
				continue;
			}


			//top-left inside and area smaller
			if ( brightSquares[i].y > brightSquares[j].y - SQUARETHRESHOLD &&
				brightSquares[i].x > brightSquares[j].x - SQUARETHRESHOLD &&
				brightSquares[i].height * brightSquares[i].width < brightSquares[j].height * brightSquares[j].width &&
				abs( brightSquares[i].y + brightSquares[i].height - brightSquares[j].y -brightSquares[j].height ) < SQUARETHRESHOLD + 20 &&
				abs( brightSquares[i].x + brightSquares[i].width - brightSquares[j].x -brightSquares[j].width ) < SQUARETHRESHOLD + 20 )
			{
				deleteme.push_back(i);
				continue;
			}

			//bottom-right inside and area smaller
			if ( abs( brightSquares[i].y - brightSquares[j].y ) < SQUARETHRESHOLD + 20 &&
				abs( brightSquares[i].x - brightSquares[j].x ) < SQUARETHRESHOLD + 20 &&
				brightSquares[i].height * brightSquares[i].width < brightSquares[j].height * brightSquares[j].width &&
				brightSquares[i].y + brightSquares[i].height < brightSquares[j].y +brightSquares[j].height + SQUARETHRESHOLD &&
				brightSquares[i].x + brightSquares[i].width < brightSquares[j].x +brightSquares[j].width + SQUARETHRESHOLD )
			{
				deleteme.push_back(i);
				continue;
			}
		}

		// remove duplicates
	  std::sort(deleteme.begin(), deleteme.end());
	  deleteme.erase(std::unique(deleteme.begin(), deleteme.end()), deleteme.end());

		for (int k = deleteme.size()-1; k >=0 ; --k)
		{
			brightSquares.erase(brightSquares.begin()+deleteme[k]);
		}
		//erase vectpr
		deleteme.clear();
	}

	for (int j = brightSquares.size()-1; j >=0 ; --j)
	{
		// if i belongs to other (j) with some margin delete and break
		for (int i = 0; i < j ; ++i)
		{
			//similar top-left and bottom right
			if ( abs( brightSquares[i].y - brightSquares[j].y ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].x - brightSquares[j].x ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].y + brightSquares[i].height - brightSquares[j].y -brightSquares[j].height ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].x + brightSquares[i].width - brightSquares[j].x -brightSquares[j].width ) < SQUARETHRESHOLD )
			{
				deleteme.push_back(i);
				continue;
			}


			//similar top-left and bottom-right inside(i)
			if ( abs( brightSquares[i].y - brightSquares[j].y ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].x - brightSquares[j].x ) < SQUARETHRESHOLD &&
				brightSquares[i].y + brightSquares[i].height < brightSquares[j].y +brightSquares[j].height + SQUARETHRESHOLD &&
				brightSquares[i].x + brightSquares[i].width < brightSquares[j].x +brightSquares[j].width + SQUARETHRESHOLD )
			{
				deleteme.push_back(i);
				continue;
			}

			//top-left inside and bottom-right similar
			if ( brightSquares[i].y > brightSquares[j].y - SQUARETHRESHOLD &&
				brightSquares[i].x > brightSquares[j].x - SQUARETHRESHOLD &&
				abs( brightSquares[i].y + brightSquares[i].height - brightSquares[j].y -brightSquares[j].height ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].x + brightSquares[i].width - brightSquares[j].x -brightSquares[j].width ) < SQUARETHRESHOLD )
			{
				deleteme.push_back(i);
				continue;
			}

			//top-left inside and bottom-right inside
			if ( brightSquares[i].y > brightSquares[j].y - SQUARETHRESHOLD &&
				brightSquares[i].x > brightSquares[j].x - SQUARETHRESHOLD &&
				brightSquares[i].y + brightSquares[i].height < brightSquares[j].y +brightSquares[j].height + SQUARETHRESHOLD &&
				brightSquares[i].x + brightSquares[i].width < brightSquares[j].x +brightSquares[j].width + SQUARETHRESHOLD )
			{
				deleteme.push_back(i);
				continue;
			}

			//top-left inside and area smaller
			if ( brightSquares[i].y > brightSquares[j].y - SQUARETHRESHOLD &&
				brightSquares[i].x > brightSquares[j].x - SQUARETHRESHOLD &&
				brightSquares[i].height * brightSquares[i].width < brightSquares[j].height * brightSquares[j].width &&
				abs( brightSquares[i].y + brightSquares[i].height - brightSquares[j].y -brightSquares[j].height ) < SQUARETHRESHOLD + 20 &&
				abs( brightSquares[i].x + brightSquares[i].width - brightSquares[j].x -brightSquares[j].width ) < SQUARETHRESHOLD +20)
			{
				deleteme.push_back(i);
				continue;
			}

			//bottom-right inside and area smaller
			if ( abs( brightSquares[i].y - brightSquares[j].y ) < SQUARETHRESHOLD + 20 &&
				abs( brightSquares[i].x - brightSquares[j].x ) < SQUARETHRESHOLD + 20 &&
				brightSquares[i].height*brightSquares[i].width < brightSquares[j].height*brightSquares[j].width &&
				brightSquares[i].y + brightSquares[i].height < brightSquares[j].y +brightSquares[j].height + SQUARETHRESHOLD &&
				brightSquares[i].x + brightSquares[i].width < brightSquares[j].x +brightSquares[j].width + SQUARETHRESHOLD )
			{
				deleteme.push_back(i);
				continue;
			}
		}
		// remove duplicates
		std::sort(deleteme.begin(), deleteme.end());
		deleteme.erase(std::unique(deleteme.begin(), deleteme.end()), deleteme.end());

		for (int k = deleteme.size()-1; k >=0 ; --k)
		{
			brightSquares.erase(brightSquares.begin()+deleteme[k]);
		}
		//erase vectpr
		deleteme.clear();
	}

	//erase vectpr
	deleteme.clear();	

	Mat tmpCol;
	bool cont = false;
	// check what is inside squares
	for (int i = 0; i < brightSquares.size(); ++i)
	{
		cont = false;
		extractRegion(original, tmp, brightSquares[i].x, brightSquares[i].y, brightSquares[i].width);//original
		extractRegion(darken, tmpCol, brightSquares[i].x, brightSquares[i].y, brightSquares[i].width);//original
		// threshold(tmp, tmp, 0, 255, THRESH_BINARY+ THRESH_OTSU);
		std::vector<double> v = nLvlTrsh(tmp, tmp);

		if ((v[0] > 0.1 && v[1] >0.1 && v[2]<0.1) || (v[3]+v[4]+v[5]>0.72) ||
			v[5]>0.55 ||
			v[0]==0 || v[1]==0 || v[2]==0 || v[3]==0 || v[4]==0 || v[5]==0)
		{
			//not a circle
			deleteme.push_back(i);
			continue;
		}

		//dartboards are most common in rows -- probability prior
		for (int l = 0; l < brightSpots.size(); ++l)
		{
			if ( abs(brightSquares[i].y+(0.5*brightSquares[i].width) - brightSpots[l].x) > BOARDHORIZONT )
			{
				deleteme.push_back(i);
				cont = true;
				break;
			}
		}

		if(cont)
			continue;

		for (int g = 0; g < 6; ++g)
		{
			cout << "La Isla Bonita: " << v[g] << endl;
		}
		cout << "=================================" << endl;


	}
	for (int k = deleteme.size()-1; k >=0 ; --k)
	{
		brightSquares.erase(brightSquares.begin()+deleteme[k]);
	}

	// print detected squares
	for (int i = 0; i < brightSquares.size(); ++i)
	{
		rectangle(frame, Point(brightSquares[i].x, brightSquares[i].y), Point(brightSquares[i].x + brightSquares[i].width, brightSquares[i].y + brightSquares[i].height), Scalar( 0, 255, 255 ), 2);
	}

	//-- Save what you got
	// imshow("output",frame);
	// waitKey();
	imwrite( "output.jpg", frame );
	cout << "___________________________\nEXPECTED EOF\n---------------------------" << endl;

}
