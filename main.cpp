#include "MyFunction.h"



TransParm2 LineMod(Mat &srcImgL, Mat &srcImgR, float(*tab)[8], int resizeFactor)
{
	//load the maskImage's information
	Mat maskImgLInfo5, maskImgRInfo5;
	int maskImgLCol5, maskImgLRow5, maskImgRCol5, maskImgRRow5;
	FileStorage fs5("Fifth_maskImageInfo.xml", FileStorage::READ);
	fs5["maskImgLInfo"] >> maskImgLInfo5;
	fs5["maskImgRInfo"] >> maskImgRInfo5;
	fs5["maskImgLCol"] >> maskImgLCol5;
	fs5["maskImgLRow"] >> maskImgLRow5;
	fs5["maskImgRCol"] >> maskImgRCol5;
	fs5["maskImgRRow"] >> maskImgRRow5;

	Mat maskImgLInfo4, maskImgRInfo4;
	int maskImgLCol4, maskImgLRow4, maskImgRCol4, maskImgRRow4;
	FileStorage fs4("Forth_maskImageInfo.xml", FileStorage::READ);
	fs4["maskImgLInfo"] >> maskImgLInfo4;
	fs4["maskImgRInfo"] >> maskImgRInfo4;
	fs4["maskImgLCol"] >> maskImgLCol4;
	fs4["maskImgLRow"] >> maskImgLRow4;
	fs4["maskImgRCol"] >> maskImgRCol4;
	fs4["maskImgRRow"] >> maskImgRRow4;

	double start = double(getTickCount());
	vector<FastMatchTrans> transCoarseL = CoarseShapeMatch(srcImgL, resizeFactor,
		maskImgLInfo5,maskImgLCol5, maskImgLRow5, tab);
	vector<FastMatchTrans> transCoarseR = CoarseShapeMatch(srcImgR, resizeFactor,
		maskImgRInfo5, maskImgRCol5, maskImgRRow5, tab);
	double time1 = double(getTickCount());
	cout << "CoarseShapeMatch's time is:" << (time1 - start) * 1000 / getTickFrequency() << " ms" << endl;

	vector<FastMatchTrans> expandTransNetL = ExpandTransNet(transCoarseL);
	vector<FastMatchTrans> expandTransNetR = ExpandTransNet(transCoarseR);

	vector<FastMatchTrans> transRefineL = RefineShapeMatch(expandTransNetL, 1, srcImgL,
		resizeFactor - 1, maskImgLInfo4, maskImgLCol4, maskImgLRow4, tab);
	vector<FastMatchTrans> transRefineR = RefineShapeMatch(expandTransNetR, 1, srcImgR,
		resizeFactor - 1, maskImgRInfo4, maskImgRCol4, maskImgRRow4, tab);

	double time2 = double(getTickCount());
	cout << "first RefineShapeMatch's time is:" << (time2 - time1) * 1000 / getTickFrequency() << " ms" << endl;

#if drawResult==1
	Mat resizeSrcImageL, resizeSrcImageR;
	resize(srcImgL, resizeSrcImageL, Size(srcImgL.cols / pow(2, resizeFactor - 1),
		srcImgL.rows / pow(2, resizeFactor - 1)));
	resize(srcImgR, resizeSrcImageR, Size(srcImgR.cols / pow(2, resizeFactor - 1),
		srcImgR.rows / pow(2, resizeFactor - 1)));

	Mat resizeSrcImageLBGR, resizeSrcImageRBGR;
	cvtColor(resizeSrcImageL, resizeSrcImageLBGR, CV_GRAY2BGR);
	cvtColor(resizeSrcImageR, resizeSrcImageRBGR, CV_GRAY2BGR);

	Point2f bestLocationL, bestLocationR;
	float bestThetaL, bestThetaR;
	bestLocationL = Point2f(transRefineL[0].x, transRefineL[0].y);
	bestLocationR = Point2f(transRefineR[0].x, transRefineR[0].y);
	bestThetaL = transRefineL[0].theta;
	bestThetaR = transRefineR[0].theta;

	for (int i = 0; i < maskImgLInfo4.cols; i++)
	{
		float mx = maskImgLInfo4.at<Vec4f>(0, i)[0] - maskImgLCol4 / 2;
		float my = maskImgLInfo4.at<Vec4f>(0, i)[1] - maskImgLRow4 / 2;
		float rsxl = cos(bestThetaL)*mx - sin(bestThetaL)*my;
		float rsyl = sin(bestThetaL)*mx + cos(bestThetaL)*my;
		float x = bestLocationL.x + rsxl;
		float y = bestLocationL.y + rsyl;
		Point2f pt(x, y);
		if (pt.x > 0 && pt.x < resizeSrcImageL.cols && pt.y>0 && pt.y < resizeSrcImageL.rows)
		{
			resizeSrcImageLBGR.at<Vec3b>(pt)[0] = 0;
			resizeSrcImageLBGR.at<Vec3b>(pt)[1] = 0;
			resizeSrcImageLBGR.at<Vec3b>(pt)[2] = 255;
		}
	}

	for (int i = 0; i < maskImgRInfo4.cols; i++)
	{
		float mx = maskImgRInfo4.at<Vec4f>(0, i)[0] - maskImgRCol4 / 2;
		float my = maskImgRInfo4.at<Vec4f>(0, i)[1] - maskImgRRow4 / 2;
		float rsxl = cos(bestThetaR)*mx - sin(bestThetaR)*my;
		float rsyl = sin(bestThetaR)*mx + cos(bestThetaR)*my;
		float x = bestLocationR.x + (float)(rsxl);
		float y = bestLocationR.y + (float)(rsyl);
		Point2f pt(x, y);
		if (pt.x > 0 && pt.x < resizeSrcImageR.cols && pt.y>0 && pt.y < resizeSrcImageR.rows)
		{
			resizeSrcImageRBGR.at<Vec3b>(pt)[0] = 0;
			resizeSrcImageRBGR.at<Vec3b>(pt)[1] = 0;
			resizeSrcImageRBGR.at<Vec3b>(pt)[2] = 255;
		}
	}
#endif

	TransParm2 result;
	result.bestThetaL = transRefineL[0].theta;
	result.bestThetaR = transRefineR[0].theta;
	result.bestLocationL = Point2f(transRefineL[0].x, transRefineL[0].y)*pow(2, resizeFactor - 1);
	result.bestLocationR = Point2f(transRefineR[0].x, transRefineR[0].y)*pow(2, resizeFactor - 1);
	return result;
}



void test()
{
	Mat input = imread("maskImageL.bmp", 0);
	vector<Feature>resultCFs;
	Mat xgrad, ygrad;
	Sobel(input, xgrad, CV_32F, 1, 0, 3);
	Sobel(input, ygrad, CV_32F, 0, 1, 3);

	Mat result = Mat::zeros(input.size(), CV_8UC3);

	for (int row = 0; row < result.rows; row++)
	{
		for (int col = 0; col < result.cols; col++)
		{
			result.at<Vec3b>(Point2f(col, row))[0] = 255;
			result.at<Vec3b>(Point2f(col, row))[1] = 255;
			result.at<Vec3b>(Point2f(col, row))[2] = 255;
		}
	}

	float threshold = 300;
	for (int row = 1; row < input.rows - 1; row++)
	{
		for (int col = 1; col < input.cols - 1; col++)
		{
			Point2f pt = Point2f(col, row);
			float xg = xgrad.at<float>(pt);
			float yg = ygrad.at<float>(pt);
			float margin = sqrt(pow(xg, 2) + pow(yg, 2));
			if (margin > threshold)
			{
				Feature tempF;
				tempF.x = pt.x;
				tempF.y = pt.y;
				tempF.margin = margin;
				tempF.orientation = atan2(abs(yg), xg);
				resultCFs.push_back(tempF);



				for (int row_s = -1; row_s <= 1; row_s++)
				{
					for (int col_s = -1; col_s <= 1; col_s++)
					{
						result.at<Vec3b>(pt + Point2f(col_s, row_s))[0] = 0;
						result.at<Vec3b>(pt + Point2f(col_s, row_s))[1] = 0;
						result.at<Vec3b>(pt + Point2f(col_s, row_s))[2] = 255;
					}
				}
			}
		}
	}

	imwrite("mask_contours.bmp", result);
	cout << "pause" << endl;
}

int main()
{
	test();
	//Mat maskImgL = imread("maskImageL.bmp", 0);
	//Mat maskImgR = imread("maskImageR.bmp", 0);
	//saveMaskImgInfo(maskImgL, maskImgR, 100, 0);

	/*Establish lookup table*/
	//float Table[256 - 1][8];//查找表
	//float(*tab)[8] = EstablishLookupTable(Table);

	//Mat srcImgL = imread("E:\\数据集\\1-18数据集\\位置不变\\BD-1\\L\\T-L-6.bmp", 0);
	//Mat srcImgR = imread("E:\\数据集\\1-18数据集\\位置不变\\BD-1\\R\\T-R-6.bmp", 0);

	//double start = double(getTickCount());
	//int highLevel = 5;
	//TransParm2 transResult = LineMod(srcImgL, srcImgR, tab, highLevel);
	////TransParm2 transResult = LineMod(srcImgL, srcImgR, tab, highLevel);
	//double duration_ms = (double(getTickCount()) - start) * 1000 / getTickFrequency();
	//cout << "total time is:" << duration_ms << " ms" << endl;

#if drawPicInSrc==1
	Mat maskImgLInfo, maskImgRInfo;
	int maskImgLCol, maskImgLRow, maskImgRCol, maskImgRRow;
	FileStorage fs("Origin_maskImageInfo.xml", FileStorage::READ);
	fs["maskImgLInfo"] >> maskImgLInfo;
	fs["maskImgRInfo"] >> maskImgRInfo;
	fs["maskImgLCol"] >> maskImgLCol;
	fs["maskImgLRow"] >> maskImgLRow;
	fs["maskImgRCol"] >> maskImgRCol;
	fs["maskImgRRow"] >> maskImgRRow;

	Mat srcImageLBGR, srcImageRBGR;
	cvtColor(srcImgL, srcImageLBGR, CV_GRAY2BGR);
	cvtColor(srcImgR, srcImageRBGR, CV_GRAY2BGR);

	for (int i = 0; i < maskImgLInfo.cols; i++)
	{
		float mx = maskImgLInfo.at<Vec4f>(0, i)[0] - maskImgLCol / 2;
		float my = maskImgLInfo.at<Vec4f>(0, i)[1] - maskImgLRow / 2;
		float rsxl = cos(transResult.bestThetaL)*mx - sin(transResult.bestThetaL)*my;
		float rsyl = sin(transResult.bestThetaL)*mx + cos(transResult.bestThetaL)*my;
		float x = transResult.bestLocationL.x + rsxl;
		float y = transResult.bestLocationL.y + rsyl;
		Point2f pt(x, y);
		if (pt.x > 0 && pt.x < srcImgL.cols && pt.y>0 && pt.y < srcImgL.rows)
		{
			srcImageLBGR.at<Vec3b>(pt)[0] = 0;
			srcImageLBGR.at<Vec3b>(pt)[1] = 0;
			srcImageLBGR.at<Vec3b>(pt)[2] = 255;
		}
	}

	for (int i = 0; i < maskImgRInfo.cols; i++)
	{
		float mx = maskImgRInfo.at<Vec4f>(0, i)[0] - maskImgRCol / 2;
		float my = maskImgRInfo.at<Vec4f>(0, i)[1] - maskImgRRow / 2;
		float rsxl = cos(transResult.bestThetaR)*mx - sin(transResult.bestThetaR)*my;
		float rsyl = sin(transResult.bestThetaR)*mx + cos(transResult.bestThetaR)*my;
		float x = transResult.bestLocationR.x + rsxl;
		float y = transResult.bestLocationR.y + rsyl;
		Point2f pt(x, y);
		if (pt.x > 0 && pt.x < srcImgR.cols && pt.y>0 && pt.y < srcImgR.rows)
		{
			srcImageRBGR.at<Vec3b>(pt)[0] = 0;
			srcImageRBGR.at<Vec3b>(pt)[1] = 0;
			srcImageRBGR.at<Vec3b>(pt)[2] = 255;
		}
	}
#endif

	cout << "pause" << endl;
}
