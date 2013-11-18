#ifndef FILTERS_HPP
#define FILTERS_HPP

#include "opencv.hpp"

#define IMGTHRESHOLD 50
#define HOUGHDETECTTRESHOLD 70

//line
//5 degrees is 0,087
#define DTH 9 //delta angle * 100
#define LINETHRESHOLD 220 //235
#define LINEYDIM 314

//circle
#define RMIN 5
#define RMAX 150//75
#define CIRCLETHRESHOLD 210//220

#define HOMOGENITYTHRS 40
#define HOMOGENITYCOND 60  // more than 80% in confidence interval +- 20 in %'s'

#define SHOW 0

void sharpen(cv::Mat &input, int size, cv::Mat &out);

void mexHat(cv::Mat &input, cv::Mat &out);
//line detect
void sobel(const cv::Mat& image, cv::Mat& xDeriv, cv::Mat& yDeriv, cv::Mat& grad, cv::Mat& arc);

std::vector<std::vector<int> > detectLines(const cv::Mat& grad, const cv::Mat& arc, cv::Mat& out);

std::vector<std::vector<std::vector<int> > > detectCircles(const cv::Mat& grad, const cv::Mat& arc, cv::Mat& out);

//extract square region with given top left corner and side length
void extractRegion(const cv::Mat& input, cv::Mat& output, int x, int y, int a);

bool checkHomogenity(const cv::Mat& input);

#endif