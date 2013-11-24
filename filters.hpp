#ifndef FILTERS_HPP
#define FILTERS_HPP

#include "opencv.hpp"

#define IMGTHRESHOLD 50
#define HOUGHDETECTTRESHOLD 190

#define DTH 9 //delta angle * 100
#define LINETHRESHOLD 220
#define LINEYDIM 314

//circle
#define RMIN 5
#define RMAX 175
#define CIRCLETHRESHOLD 220
#define CIRCLEROWS 70
#define CIRCLECOLS 100

#define HOMOGENITYTHRS 40
#define HOMOGENITYCOND 55 // more than 80% in confidence interval +- 20 in %'s'

// divergence of max value in circle detection
#define RADDIVER 1
#define PROXIMITY 51
#define DISCARDRADIUS 10

//maximum number of circles per image
#define MAXCIRCNO 3

#define SQUARETHRESHOLD 20

#define BOARDHORIZONT 70

#define DELETECIRCLE 65

#define SHOW 0

void sharpen(cv::Mat &input, int size, cv::Mat &out);

void mexHat(cv::Mat &input, cv::Mat &out);

//line detect
void sobel(const cv::Mat& image, cv::Mat& xDeriv, cv::Mat& yDeriv, cv::Mat& grad, cv::Mat& arc);

std::vector<std::vector<int> > detectLines(const cv::Mat& grad, const cv::Mat& arc, cv::Mat& out);

std::vector<std::vector<std::vector<int> > > detectCircles(const cv::Mat& grad, const cv::Mat& arc, cv::Mat& out, int iteration);

//extract square region with given top left corner and side length
void extractRegion(const cv::Mat& input, cv::Mat& output, int x, int y, int a);

bool checkHomogenity(const cv::Mat& input);

std::vector<double> nLvlTrsh( const cv::Mat& input, cv::Mat& output );

#endif