#pragma once
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include <opencv2/highgui/highgui_c.h> 
#include <fstream>
#include <string>
#include <vector>
#include<iostream>
#include <iterator>
#include <sstream>	
#include "stdio.h"
#include "stdlib.h"

using namespace std;
using namespace cv;

#define drawResult 0
#define drawPicInSrc 0

static constexpr const double PI = 3.141592636;

struct Gradient
{
	Point2f pt;
	float theta;
	float margin;
};

struct TransParm
{
	Point2f bestLocation;
	float bestTheta;
};

struct TransParm2
{
	Point2f bestLocationL;
	Point2f bestLocationR;
	float bestThetaL;
	float bestThetaR;
};


struct Translation
{
	float x;
	float y;
};

struct Rotation
{
	float theta;
};

struct FastMatchTrans
{
	float x;
	float y;
	float theta;
};

struct GOL
{
	vector<Point2f>location;
	vector<uchar>orientationList;
};

struct PRMaps
{
	Mat PRMap0;
	Mat PRMap1;
	Mat PRMap2;
	Mat PRMap3;
	Mat PRMap4;
	Mat PRMap5;
	Mat PRMap6;
	Mat PRMap7;
};

struct Feature
{
	int x;
	int y;
	float orientation;
	float margin;
};


int  **Max(int **arr, int n, int m);

uchar CalOrientationCode(float theta);

vector<FastMatchTrans> RefineShapeMatch(vector<FastMatchTrans> coarseTransNet,
	int outputListNum, Mat &srcImg, int resizeFactor, Mat &maskImgInfo,
	float maskImgCol, float maskImgRow, float(*tab)[8]);

float(*EstablishLookupTable(float(&Map)[256 - 1][8]))[8];

void saveMaskImgInfo(Mat &maskImgL, Mat &maskImgR, int maskFeatureNum, int resizeFactor);

vector<FastMatchTrans> CoarseShapeMatch(Mat &srcImg, int resizeFactor,
	Mat &maskImgInfo, float maskImgCol, float maskImgRow, float(*tab)[8]);

void ConstructTransNet(vector<Translation>&net, int searchCol, int searchRow);

void ConstructRotateNet(vector<Rotation>&net, float minAngle, float maxAngle, float step);

vector<FastMatchTrans>ExpandTransNet(vector<FastMatchTrans>& firstTrans);
