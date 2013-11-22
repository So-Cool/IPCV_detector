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
	std::vector<Rect> faces, faces1, faces2, brightSquares;
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
	if(SHOW) imshow("Dark", darken);
	if(SHOW) waitKey();

	Mat brighten;
	cv::normalize(dark, brighten, 0, 255, cv::NORM_MINMAX);
	if(SHOW) imshow("Bright", brighten);
	if(SHOW) waitKey();

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
	mexHat(temp8Bit, temp8Bit);
	if(SHOW) imshow("CRC & MexHat", temp8Bit);
	if(SHOW) waitKey();

	//choose the coordinates of brightest points
	int rowsmax = 0;
	int colsmax = 0;
	int radmax = 0;
	int vmax = 0;
	int tempmax = 0;
	bool notfound = true;


    // providing a negative number will create a filled circle
    int thickness = 2;
	// Parameters for circle
    cv::Scalar redColour(255, 0, 0);
    // 8-connected line
    int linetype = 8; 

	// int lol = 0;

	std::vector<cv::Point> brightSpots;
	std::vector<int> brightR;
	std::vector<int> brightValue;
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

				for (int k = RMIN; k < RMAX; ++k)
				{
					if ( roundShapes[i][j][k] > vmax)
					{
						vmax = roundShapes[i][j][k];
						radmax = k;
						// ++lol;
					}
				}

				if (radmax > DISCARDRADIUS)
				{

					// if list empty put first element
					if (brightSpots.size() == 0 && radmax*2<imageCols && radmax*2<imageRows
						&& i+radmax<imageRows && i-radmax>0 &&  j+radmax<imageCols && j-radmax>0)
					{
						// cout << "dodaje bo empty: " << i << " " << j << vmax << endl;
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
						    // myvector.erase (myvector.begin()+5);
						    notfound = true;
							if  ( abs(brightSpots[l].x - i ) < PROXIMITY &&
									abs( brightSpots[l].y - j ) < PROXIMITY)
							{
								if( roundShapes[i][j][radmax] > roundShapes[brightSpots[l].x][brightSpots[l].y][brightR[l]]
									&& radmax*2<imageCols && radmax*2<imageRows
									&& i+radmax<imageRows && i-radmax>0 &&  j+radmax<imageCols && j-radmax>0 )
								{

									// cout << abs(brightSpots[l].x - i ) << "  " << abs(brightSpots[l].y - j ) << endl;
									//swap
											// pamietaj o R
									// cout << "L before: " << brightSpots.size() << endl;
									brightSpots.erase(brightSpots.begin()+ l);
									brightR.erase(brightR.begin()+ l);
									brightValue.erase(brightValue.begin()+ l);
									// cout << "L after: " << brightSpots.size() << endl;


									// cout << " Found similar " << endl;
									// brightSpots.push_back(cv::Point(i, j));
									// brightR.push_back(radmax);
									// brightValue.push_back(vmax);

									notfound=true;
									break;
								}
								else
								{
									notfound = false;
									break;
								}
							}
							// else 
							// 	notfound = true ;
						}

						// add new if not fount
						if (notfound && radmax*2<imageCols && radmax*2<imageRows
							&& i+radmax<imageRows && i-radmax>0 &&  j+radmax<imageCols && j-radmax>0)
						{
							// cout << "dodaje: " << i << " " << j << endl;
							brightSpots.push_back(cv::Point(i, j));
							brightR.push_back(radmax);
							brightValue.push_back(vmax);
						}

					}
				}
			}
		}
	}


	// print circles
	for (int i = 0; i < brightSpots.size(); ++i)
	{
		cout << "CIRC: " << brightSpots[i].x << " " << brightSpots[i].y << " " << brightR[i] << endl;
		// brightValue.push_back(vmax);
	    int radius = brightR[i];
		cv::Point center( brightSpots[i].y, brightSpots[i].x );
		cv::circle ( frame , center , radius , redColour , thickness , linetype );
	}




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
	logo_cascade.detectMultiScale( gaussianB5, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) ); //blurred
	logo_cascade.detectMultiScale( gaussianB3, faces1, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) ); //blurred
	logo_cascade.detectMultiScale( gaussianB1, faces2, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) ); //blurred
	std::cout << faces.size() + faces1.size() + faces2.size() << std::endl;

	// faces.insert( faces.end(), faces1.begin(), faces1.end() );
	// faces.insert( faces.end(), faces2.begin(), faces2.end() );
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
		//circle in a square? Don't print it!
		// if (brightSpots.size() == 1)
		// {
		// 	if (brightSpots[0].y > faces[i].x && brightSpots[0].y < faces[i].x+faces[i].width
		// 		&& brightSpots[0].x > faces[i].y && brightSpots[0].x < faces[i].y+faces[i].height )
		// 	{
		// 		continue;
		// 	}
		// }
		// else
		// {
			for (int j = 0; j < brightSpots.size(); ++j)
			{
				if (brightSpots[j].y > faces[i].x && brightSpots[j].y < faces[i].x+faces[i].width
					&& brightSpots[j].x > faces[i].y && brightSpots[j].x < faces[i].y+faces[i].height )
				{
					breakable = true;
					break;
				}

				// if one of vertices of square is in the circle delete it
				// REMEMBER THAT FIRST DELETE REDUNDANT CIRCLES:
				//top-left
				if (faces[i].x < brightSpots[j].y + brightR[j] && faces[i].x > brightSpots[j].y - brightR[j] && //horizontal lining
					faces[i].y > brightSpots[j].x - brightR[j] && faces[i].y < brightSpots[j].x + brightR[j] ) //vertical lining
				{
					breakable = true;
					break;
				}
				//top-right
				else if (faces[i].x < brightSpots[j].y + brightR[j] && faces[i].x > brightSpots[j].y - brightR[j] && //horizontal lining
					faces[i].y + faces[i].width > brightSpots[j].x - brightR[j] && faces[i].y + faces[i].width < brightSpots[j].x + brightR[j] ) //vertical lining
				{
					breakable = true;
					break;
				}
				//bottom-right
				else if (faces[i].x + faces[i].height < brightSpots[j].y + brightR[j] && faces[i].x + faces[i].height > brightSpots[j].y - brightR[j] && //horizontal lining
					faces[i].y + faces[i].width > brightSpots[j].x - brightR[j] && faces[i].y + faces[i].width < brightSpots[j].x + brightR[j] ) //vertical lining
				{
					breakable = true;
					break;
				}
				//bottom-left
				else if (faces[i].x + faces[i].height < brightSpots[j].y + brightR[j] && faces[i].x + faces[i].height > brightSpots[j].y - brightR[j] && //horizontal lining
					faces[i].y > brightSpots[j].x - brightR[j] && faces[i].y < brightSpots[j].x + brightR[j] ) //vertical lining
				{
					breakable = true;
					break;
				}

			}
			if (breakable)
			{
				continue;
			}
		// }







		// Canny(tmp, tmp, 50, 200, 3);
		// imshow("ext", tmp);
		// waitKey();
		sobel(tmp, line_xDeriv, line_yDeriv, line_grad, line_arc);//original
		// detectLines(line_grad, line_arc, tmp);
		for (int x = 0; x < tmp.rows; ++x)
		{
			for (int y = 0; y < tmp.cols; ++y)
			{
				if (tmp.at<uchar>(x,y) >160)//220 pretty good
				{
					tmp.at<uchar>(x,y) = 255  ;
				}
				else {
					tmp.at<uchar>(x,y) = 0  ;
				}
			}
		}
		detectCircles(line_grad, line_arc, tmp);
		for (int x = 0; x < tmp.rows; ++x)
		{
			for (int y = 0; y < tmp.cols; ++y)
			{
				if (tmp.at<uchar>(x,y) < CIRCLETHRESHOLD)//220 pretty good
				{
					tmp.at<uchar>(x,y) = 0  ;
				}
			}
		}
		mexHat(tmp, tmp);
		// imshow("extLine", tmp);
		// waitKey();

		brightSquares.push_back(faces[i]);
		// rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}
	for( int i = 0; i < faces1.size(); i++ )
	{
		breakable = false;
		extractRegion(original, tmp, faces1[i].x, faces1[i].y, faces1[i].width);
		if(checkHomogenity(tmp))//size matters
			continue;
		//circle in a square? Don't print it!
		// if (brightSpots.size() == 1)
		// {
		// 	if (brightSpots[0].y > faces1[i].x && brightSpots[0].y < faces1[i].x+faces1[i].width
		// 		&& brightSpots[0].x > faces1[i].y && brightSpots[0].x < faces1[i].y+faces1[i].height )
		// 	{
		// 		continue;
		// 	}
		// }
		// else
		// {
			for (int j = 0; j < brightSpots.size(); ++j)
			{
				if (brightSpots[j].y > faces1[i].x && brightSpots[j].y < faces1[i].x+faces1[i].width
					&& brightSpots[j].x > faces1[i].y && brightSpots[j].x < faces1[i].y+faces1[i].height )
				{
					breakable = true;
					break;
				}


				// if one of vertices of square is in the circle delete it
				// REMEMBER THAT FIRST DELETE REDUNDANT CIRCLES:
				//top-left
				if (faces1[i].x < brightSpots[j].y + brightR[j] && faces1[i].x > brightSpots[j].y - brightR[j] && //horizontal lining
					faces1[i].y > brightSpots[j].x - brightR[j] && faces1[i].y < brightSpots[j].x + brightR[j] ) //vertical lining
				{
					breakable = true;
					break;
				}
				//top-right
				else if (faces1[i].x < brightSpots[j].y + brightR[j] && faces1[i].x > brightSpots[j].y - brightR[j] && //horizontal lining
					faces1[i].y + faces1[i].width > brightSpots[j].x - brightR[j] && faces1[i].y + faces1[i].width < brightSpots[j].x + brightR[j] ) //vertical lining
				{
					breakable = true;
					break;
				}
				//bottom-right
				else if (faces1[i].x + faces1[i].height < brightSpots[j].y + brightR[j] && faces1[i].x + faces1[i].height > brightSpots[j].y - brightR[j] && //horizontal lining
					faces1[i].y + faces1[i].width > brightSpots[j].x - brightR[j] && faces1[i].y + faces1[i].width < brightSpots[j].x + brightR[j] ) //vertical lining
				{
					breakable = true;
					break;
				}
				//bottom-left
				else if (faces1[i].x + faces1[i].height < brightSpots[j].y + brightR[j] && faces1[i].x + faces1[i].height > brightSpots[j].y - brightR[j] && //horizontal lining
					faces1[i].y > brightSpots[j].x - brightR[j] && faces1[i].y < brightSpots[j].x + brightR[j] ) //vertical lining
				{
					breakable = true;
					break;
				}


			}
			if (breakable)
			{
				continue;
			}
		// }





		Canny(tmp, tmp, 50, 200, 3);
		// imshow("ext", tmp);
		// waitKey();
		sobel(tmp, line_xDeriv, line_yDeriv, line_grad, line_arc);//original
		detectLines(line_grad, line_arc, tmp);
		// mexHat(tmp, tmp);
		// imshow("extLine", tmp);
		// waitKey();

		brightSquares.push_back(faces1[i]);
		// rectangle(frame, Point(faces1[i].x, faces1[i].y), Point(faces1[i].x + faces1[i].width, faces1[i].y + faces1[i].height), Scalar( 0, 0, 255 ), 2);
	}

	//delete overlapping squares
	for (int i = 0; i < brightSquares.size(); ++i)
	{
		// if i belongs to other (j) with some margin delete and break
		for (int j = i+1; j < brightSquares.size(); ++j)
		{
			//similar top-left and bottom right
			if ( abs( brightSquares[i].x - brightSquares[j].x ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].y - brightSquares[j].y ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].x + brightSquares[i].height - brightSquares[j].x -brightSquares[j].height ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].y + brightSquares[i].width - brightSquares[j].y -brightSquares[j].width ) < SQUARETHRESHOLD )
			{
				brightSquares.erase(brightSquares.begin()+ i);
				break;
			}


			//similar top-left and bottom-right inside(i)
			if ( abs( brightSquares[i].x - brightSquares[j].x ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].y - brightSquares[j].y ) < SQUARETHRESHOLD &&
				brightSquares[i].x + brightSquares[i].height < brightSquares[j].x +brightSquares[j].height + SQUARETHRESHOLD &&
				brightSquares[i].y + brightSquares[i].width < brightSquares[j].y +brightSquares[j].width + SQUARETHRESHOLD )
			{
				brightSquares.erase(brightSquares.begin()+ i);
				break;
			}

			//top-left inside and bottom-right similar
			if ( brightSquares[i].x > brightSquares[j].x - SQUARETHRESHOLD &&
				brightSquares[i].y > brightSquares[j].y - SQUARETHRESHOLD &&
				abs( brightSquares[i].x + brightSquares[i].height - brightSquares[j].x -brightSquares[j].height ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].y + brightSquares[i].width - brightSquares[j].y -brightSquares[j].width ) < SQUARETHRESHOLD )
			{
				brightSquares.erase(brightSquares.begin()+ i);
				break;
			}

			//top-left inside and bottom-right inside
			if ( brightSquares[i].x > brightSquares[j].x - SQUARETHRESHOLD &&
				brightSquares[i].y > brightSquares[j].y - SQUARETHRESHOLD &&
				brightSquares[i].x + brightSquares[i].height < brightSquares[j].x +brightSquares[j].height + SQUARETHRESHOLD &&
				brightSquares[i].y + brightSquares[i].width < brightSquares[j].y +brightSquares[j].width + SQUARETHRESHOLD )
			{
				brightSquares.erase(brightSquares.begin()+ i);
				break;
			}









			// // one inside other
			// if ( brightSquares[i].x - brightSquares[j].x > 0 &&
			// 	brightSquares[i].y - brightSquares[j].y > 0 &&
			// 	abs( brightSquares[i].width -brightSquares[j].width ) < SQUARETHRESHOLD+20 &&
			// 	abs( brightSquares[i].height  -brightSquares[j].height ) < SQUARETHRESHOLD+20 )
			// {
			// 	brightSquares.erase(brightSquares.begin()+ i);
			// 	break;
			// }

			// if ( brightSquares[i].x - brightSquares[j].x > 0 &&
			// 	brightSquares[i].y - brightSquares[j].y > 0 &&
			// 	abs( brightSquares[i].x + brightSquares[i].width - brightSquares[j].x -brightSquares[j].width ) < SQUARETHRESHOLD &&
			// 	abs( brightSquares[i].y + brightSquares[i].height - brightSquares[j].y -brightSquares[j].height ) < SQUARETHRESHOLD )
			// {
			// 	brightSquares.erase(brightSquares.begin()+ i);
			// 	break;
			// }


			// if ( brightSquares[i].x - brightSquares[j].x < 0 &&
			// 	brightSquares[i].y - brightSquares[j].y < 0 &&
			// 	abs( brightSquares[i].x + brightSquares[i].width - brightSquares[j].x -brightSquares[j].width ) < SQUARETHRESHOLD &&
			// 	abs( brightSquares[i].y + brightSquares[i].height - brightSquares[j].y -brightSquares[j].height ) < SQUARETHRESHOLD )
			// {
			// 	brightSquares.erase(brightSquares.begin()+ j);
			// 	break;
			// }

			// if ( brightSquares[i].x - brightSquares[j].x < 0 &&
			// 	brightSquares[i].y - brightSquares[j].y < 0 &&
			// 	abs( brightSquares[i].width -brightSquares[j].width ) < SQUARETHRESHOLD+20 &&
			// 	abs( brightSquares[i].height  -brightSquares[j].height ) < SQUARETHRESHOLD+20 )
			// {
			// 	brightSquares.erase(brightSquares.begin()+ j);
			// 	break;
			// }

		}
		// rectangle(frame, Point(brightSquares[i].x, brightSquares[i].y), Point(brightSquares[i].x + brightSquares[i].width, brightSquares[i].y + brightSquares[i].height), Scalar( 0, 255, 255 ), 2);
	}	
		for (int j = 0; j < brightSquares.size(); ++j)
	{
		// if i belongs to other (j) with some margin delete and break
		for (int i = j+1; i < brightSquares.size(); ++i)
		{
			//similar top-left and bottom right
			if ( abs( brightSquares[i].x - brightSquares[j].x ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].y - brightSquares[j].y ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].x + brightSquares[i].height - brightSquares[j].x -brightSquares[j].height ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].y + brightSquares[i].width - brightSquares[j].y -brightSquares[j].width ) < SQUARETHRESHOLD )
			{
				brightSquares.erase(brightSquares.begin()+ i);
				break;
			}


			//similar top-left and bottom-right inside(i)
			if ( abs( brightSquares[i].x - brightSquares[j].x ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].y - brightSquares[j].y ) < SQUARETHRESHOLD &&
				brightSquares[i].x + brightSquares[i].height < brightSquares[j].x +brightSquares[j].height + SQUARETHRESHOLD &&
				brightSquares[i].y + brightSquares[i].width < brightSquares[j].y +brightSquares[j].width + SQUARETHRESHOLD )
			{
				brightSquares.erase(brightSquares.begin()+ i);
				break;
			}

			//top-left inside and bottom-right similar
			if ( brightSquares[i].x > brightSquares[j].x - SQUARETHRESHOLD &&
				brightSquares[i].y > brightSquares[j].y - SQUARETHRESHOLD &&
				abs( brightSquares[i].x + brightSquares[i].height - brightSquares[j].x -brightSquares[j].height ) < SQUARETHRESHOLD &&
				abs( brightSquares[i].y + brightSquares[i].width - brightSquares[j].y -brightSquares[j].width ) < SQUARETHRESHOLD )
			{
				brightSquares.erase(brightSquares.begin()+ i);
				break;
			}

			//top-left inside and bottom-right inside
			if ( brightSquares[i].x > brightSquares[j].x - SQUARETHRESHOLD &&
				brightSquares[i].y > brightSquares[j].y - SQUARETHRESHOLD &&
				brightSquares[i].x + brightSquares[i].height < brightSquares[j].x +brightSquares[j].height + SQUARETHRESHOLD &&
				brightSquares[i].y + brightSquares[i].width < brightSquares[j].y +brightSquares[j].width + SQUARETHRESHOLD )
			{
				brightSquares.erase(brightSquares.begin()+ i);
				break;
			}
		}
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

}
