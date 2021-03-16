#include "MyFunction.h"

using namespace std;
using namespace cv;




int  **Max(int **arr, int n, int m)
{
	int **data;//定义二重指针
	data = (int **)malloc(n * sizeof(int *));//为二重指针开辟空间
	for (int i = 0; i < n; i++)
		data[i] = (int *)malloc(2 * sizeof(int));
	for (int i = 0; i < n; ++i)
	{
		int maxNum = 0;
		for (int j = 0; j < m; ++j)
		{
			if (*((int *)arr + m * i + j) > maxNum)
			{
				maxNum = *((int *)arr + m * i + j);
				data[i][0] = maxNum; data[i][1] = j;
			}
		}
	}
	return data;
}


uchar CalOrientationCode(float theta)
{
	uchar result = 0, temp = 0;
	float thetaStep = PI / 8;
	if (theta >= 0 && theta < thetaStep) temp = 1 << 0;
	if (theta >= thetaStep && theta < 2 * thetaStep)  temp = 1 << 1;
	if (theta >= 2 * thetaStep && theta < 3 * thetaStep) temp = 1 << 2;
	if (theta >= 3 * thetaStep && theta < 4 * thetaStep) temp = 1 << 3;
	if (theta >= 4 * thetaStep && theta < 5 * thetaStep) temp = 1 << 4;
	if (theta >= 5 * thetaStep && theta < 6 * thetaStep) temp = 1 << 5;
	if (theta >= 6 * thetaStep && theta < 7 * thetaStep) temp = 1 << 6;
	if (theta >= 7 * thetaStep && theta < 8 * thetaStep) temp = 1 << 7;

	if (theta >= 8 * thetaStep && theta < 9 * thetaStep)   temp = 1 << 0;
	if (theta >= 9 * thetaStep && theta < 10 * thetaStep)  temp = 1 << 1;
	if (theta >= 10 * thetaStep && theta < 11 * thetaStep) temp = 1 << 2;
	if (theta >= 11 * thetaStep && theta < 12 * thetaStep) temp = 1 << 3;
	if (theta >= 12 * thetaStep && theta < 13 * thetaStep) temp = 1 << 4;
	if (theta >= 13 * thetaStep && theta < 14 * thetaStep) temp = 1 << 5;
	if (theta >= 14 * thetaStep && theta < 15 * thetaStep) temp = 1 << 6;
	if (theta >= 15 * thetaStep && theta < 16 * thetaStep) temp = 1 << 7;

	return result |= temp;
}


/*************************************************************
Function:       GetOriginalGradientImg
Description:    计算梯度方向列表
Input:          image:待测图像
Return:         gradientOrientaList:图像梯度方向列表
**************************************************************/
Mat GetOriJ(Mat& image)
{
	Mat xgrade, ygrade;
	Sobel(image, xgrade, CV_32F, 1, 0, 3);
	Sobel(image, ygrade, CV_32F, 0, 1, 3);

	Mat J = Mat::zeros(image.size(), image.type());

	GOL gradientOrientaList;
	vector<float>thetaList;
	int featureNum = 30;
	//float distance=featureNum/
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			Point2f pt(col, row);
			float xg = xgrade.at<float>(pt);
			float yg = ygrade.at<float>(pt);
			float margin = sqrt(pow(xg, 2) + pow(yg, 2));
			if (margin > 200)
			{
				float theta = abs(atan2(yg, xg));
				uchar ori = CalOrientationCode(theta);
				gradientOrientaList.location.push_back(pt);
				gradientOrientaList.orientationList.push_back(ori);
			}
		}
	}

	for (int indexJ = 0; indexJ < gradientOrientaList.location.size(); indexJ++)
	{
		Point2f pt = Point2f(gradientOrientaList.location[indexJ]);
		J.at<uchar>(pt) |= gradientOrientaList.orientationList[indexJ];
		if (pt.x > 0 && pt.x < J.cols - 1 && pt.y>0 && pt.y < J.rows - 1)
		{
			J.at<uchar>(pt + Point2f(1, 0)) |= gradientOrientaList.orientationList[indexJ];
			J.at<uchar>(pt + Point2f(1, 1)) |= gradientOrientaList.orientationList[indexJ];
			J.at<uchar>(pt + Point2f(0, 1)) |= gradientOrientaList.orientationList[indexJ];
			J.at<uchar>(pt + Point2f(-1, 1)) |= gradientOrientaList.orientationList[indexJ];
			J.at<uchar>(pt + Point2f(-1, 0)) |= gradientOrientaList.orientationList[indexJ];
			J.at<uchar>(pt + Point2f(-1, -1)) |= gradientOrientaList.orientationList[indexJ];
			J.at<uchar>(pt + Point2f(0, -1)) |= gradientOrientaList.orientationList[indexJ];
			J.at<uchar>(pt + Point2f(1, -1)) |= gradientOrientaList.orientationList[indexJ];
		}
	}

	return J;
}


Mat CalPRMap(Mat &J, int direction)
{
	vector<uchar>bitsCode;
	Mat PRMap = Mat::zeros(J.size(), CV_32FC1);
	for (int row = 0; row < J.rows; row++)
	{
		for (int col = 0; col < J.cols; col++)
		{
			Point2f pt = Point2f(col, row);
			uchar curJValue = J.at<uchar>(pt);

			bitsCode.push_back(curJValue & 1);
			bitsCode.push_back(curJValue & 2);
			bitsCode.push_back(curJValue & 4);
			bitsCode.push_back(curJValue & 8);
			bitsCode.push_back(curJValue & 16);
			bitsCode.push_back(curJValue & 32);
			bitsCode.push_back(curJValue & 64);
			bitsCode.push_back(curJValue & 128);

			int cursor, cursorC = direction, cursorL = 0, cursorR = 0;

			while (bitsCode[cursorC - cursorL] == 0)
			{
				if (cursorC - cursorL < 0)
				{
					break;
				}
				cursorL++;
			}
			while (bitsCode[cursorC + cursorR] == 0)
			{
				if (cursorC + cursorR >= bitsCode.size())
				{
					break;
				}
				cursorR++;
			}
			cursor = cursorL >= cursorR ? cursorR : cursorL;

			cout << "row: " << row << ", col: " << col << endl;
			PRMap.at<float>(pt) = abs(cos(cursor * PI / 16));
		}
	}

	return PRMap;
}


PRMaps GetPRMaps(Mat image)
{
	PRMaps resultPRMaps;
	resultPRMaps.PRMap0 = CalPRMap(image, 0);
	resultPRMaps.PRMap1 = CalPRMap(image, 1);
	resultPRMaps.PRMap2 = CalPRMap(image, 2);
	resultPRMaps.PRMap3 = CalPRMap(image, 3);
	resultPRMaps.PRMap4 = CalPRMap(image, 4);
	resultPRMaps.PRMap5 = CalPRMap(image, 5);
	resultPRMaps.PRMap6 = CalPRMap(image, 6);
	resultPRMaps.PRMap7 = CalPRMap(image, 7);

	return resultPRMaps;
}

Mat LineMod(Mat srcImg, Mat maskImg)
{
	Mat maskImgJ = GetOriJ(maskImg);
	PRMaps maskPRMaps = GetPRMaps(maskImgJ);

	Mat srcImgJ = GetOriJ(srcImg);
	PRMaps srcPRMaps = GetPRMaps(srcImgJ);

	return srcImg;
}


class imageInfo
{
public:
	vector<Feature>featurePoints;
	Mat xgrad;
	Mat ygrad;
	vector<uchar>oris;
};


imageInfo ProcessImage(Mat& input, int featureNum)
{
	vector<Feature>resultCFs;
	Mat xgrad, ygrad;
	Sobel(input, xgrad, CV_32F, 1, 0, 3);
	Sobel(input, ygrad, CV_32F, 0, 1, 3);

	float threshold = 200;
	for (int row = 2; row < input.rows - 2; row++)
	{
		for (int col = 2; col < input.cols - 2; col++)
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
			}
		}
	}

	float distance = 5;
	int m = resultCFs.size();
	while (resultCFs.size() > featureNum)
	{
		for (int i = 0; i < m - 1; i++)
		{
			for (int j = i + 1; j < m; j++)
			{
				float curDistance = sqrt((resultCFs[i].x - resultCFs[j].x)*(resultCFs[i].x - resultCFs[j].x) +
					(resultCFs[i].y - resultCFs[j].y)*(resultCFs[i].y - resultCFs[j].y));

				if (curDistance <= distance)
				{
					resultCFs.erase(resultCFs.begin() + j);
					m--;
				}
			}
		}
		distance *= 1.2;
	}

	imageInfo result;
	result.featurePoints = resultCFs;
	result.xgrad = xgrad;
	result.ygrad = ygrad;
	return result;
}


void saveMaskImgInfo(Mat &maskImgL, Mat &maskImgR, int maskFeatureNum, int resizeFactor)
{
	vector<Feature>maskLFeatures, maskRFeatures, srcFeatures;
	imageInfo maskLInfo, maskRInfo;

	Mat resizeMaskImgL, resizeMaskImgR;
	resize(maskImgL, resizeMaskImgL, Size(maskImgL.cols / pow(2, resizeFactor),
		maskImgL.rows / pow(2, resizeFactor)));	
	resize(maskImgR, resizeMaskImgR, Size(maskImgR.cols / pow(2, resizeFactor),
		maskImgR.rows / pow(2, resizeFactor)));

	maskLInfo = ProcessImage(resizeMaskImgL, maskFeatureNum);
	maskRInfo = ProcessImage(resizeMaskImgR, maskFeatureNum);
	maskLFeatures = maskLInfo.featurePoints;
	maskRFeatures = maskRInfo.featurePoints;

	Mat maskImgLInfo(1, maskLFeatures.size(), CV_32FC4, cv::Scalar(1.0f, 0.0f, 0.0f, 0.0f));
	Mat maskImgRInfo(1, maskRFeatures.size(), CV_32FC4, cv::Scalar(1.0f, 0.0f, 0.0f, 0.0f));
	for (int i = 0; i < maskLFeatures.size(); i++)
	{
		maskImgLInfo.at<Vec4f>(0, i)[0] = (float)(maskLFeatures[i].x);
		maskImgLInfo.at<Vec4f>(0, i)[1] = (float)(maskLFeatures[i].y);
		maskImgLInfo.at<Vec4f>(0, i)[2] = (float)(maskLFeatures[i].orientation);
		maskImgLInfo.at<Vec4f>(0, i)[3] = (float)(maskLFeatures[i].margin);
	}
	for (int i = 0; i < maskRFeatures.size(); i++)
	{
		maskImgRInfo.at<Vec4f>(0, i)[0] = (float)(maskRFeatures[i].x);
		maskImgRInfo.at<Vec4f>(0, i)[1] = (float)(maskRFeatures[i].y);
		maskImgRInfo.at<Vec4f>(0, i)[2] = (float)(maskRFeatures[i].orientation);
		maskImgRInfo.at<Vec4f>(0, i)[3] = (float)(maskRFeatures[i].margin);
	}

	FileStorage fs("Origin_maskImageInfo.xml", FileStorage::WRITE);
	fs << "maskImgLInfo" << maskImgLInfo << "maskImgRInfo" << maskImgRInfo
		<< "maskImgLCol" << resizeMaskImgL.cols << "maskImgLRow" << resizeMaskImgL.rows
		<< "maskImgRCol" << resizeMaskImgR.cols << "maskImgRRow" << resizeMaskImgR.rows;
	fs.release();
}


int Paritition(float A[], int low, int high) {
	float pivot = A[low];
	while (low < high) {
		while (low < high && A[high] >= pivot) {
			--high;
		}
		A[low] = A[high];
		while (low < high && A[low] <= pivot) {
			++low;
		}
		A[high] = A[low];
	}
	A[low] = pivot;
	return low;
}
void QuickSort(float A[], int low, int high) //快排母函数
{
	if (low < high) {
		int pivot = Paritition(A, low, high);
		QuickSort(A, low, pivot - 1);
		QuickSort(A, pivot + 1, high);
	}
}
void GetSortedDirListInRegion(float A[], Mat &xgrad, Mat&ygrad, Point2f pt0, vector<Point2f>diff)
{
	float direction0 = atan2(abs(ygrad.at<float>(pt0 + diff[0])), xgrad.at<float>(pt0 + diff[0]));
	float direction1 = atan2(abs(ygrad.at<float>(pt0 + diff[1])), xgrad.at<float>(pt0 + diff[1]));
	float direction2 = atan2(abs(ygrad.at<float>(pt0 + diff[2])), xgrad.at<float>(pt0 + diff[2]));
	float direction3 = atan2(abs(ygrad.at<float>(pt0 + diff[3])), xgrad.at<float>(pt0 + diff[3]));
	float direction4 = atan2(abs(ygrad.at<float>(pt0 + diff[4])), xgrad.at<float>(pt0 + diff[4]));
	float direction5 = atan2(abs(ygrad.at<float>(pt0 + diff[5])), xgrad.at<float>(pt0 + diff[5]));
	float direction6 = atan2(abs(ygrad.at<float>(pt0 + diff[6])), xgrad.at<float>(pt0 + diff[6]));
	float direction7 = atan2(abs(ygrad.at<float>(pt0 + diff[7])), xgrad.at<float>(pt0 + diff[7]));
	float direction8 = atan2(abs(ygrad.at<float>(pt0 + diff[8])), xgrad.at<float>(pt0 + diff[8]));
	float direction9 = atan2(abs(ygrad.at<float>(pt0 + diff[9])), xgrad.at<float>(pt0 + diff[9]));
	float direction10 = atan2(abs(ygrad.at<float>(pt0 + diff[10])), xgrad.at<float>(pt0 + diff[10]));
	float direction11 = atan2(abs(ygrad.at<float>(pt0 + diff[11])), xgrad.at<float>(pt0 + diff[11]));
	float direction12 = atan2(abs(ygrad.at<float>(pt0 + diff[12])), xgrad.at<float>(pt0 + diff[12]));
	float direction13 = atan2(abs(ygrad.at<float>(pt0 + diff[13])), xgrad.at<float>(pt0 + diff[13]));
	float direction14 = atan2(abs(ygrad.at<float>(pt0 + diff[14])), xgrad.at<float>(pt0 + diff[14]));
	float direction15 = atan2(abs(ygrad.at<float>(pt0 + diff[15])), xgrad.at<float>(pt0 + diff[15]));
	float direction16 = atan2(abs(ygrad.at<float>(pt0 + diff[16])), xgrad.at<float>(pt0 + diff[16]));
	float direction17 = atan2(abs(ygrad.at<float>(pt0 + diff[17])), xgrad.at<float>(pt0 + diff[17]));
	float direction18 = atan2(abs(ygrad.at<float>(pt0 + diff[18])), xgrad.at<float>(pt0 + diff[18]));
	float direction19 = atan2(abs(ygrad.at<float>(pt0 + diff[19])), xgrad.at<float>(pt0 + diff[19]));
	float direction20 = atan2(abs(ygrad.at<float>(pt0 + diff[20])), xgrad.at<float>(pt0 + diff[20]));
	float direction21 = atan2(abs(ygrad.at<float>(pt0 + diff[21])), xgrad.at<float>(pt0 + diff[21]));
	float direction22 = atan2(abs(ygrad.at<float>(pt0 + diff[22])), xgrad.at<float>(pt0 + diff[22]));
	float direction23 = atan2(abs(ygrad.at<float>(pt0 + diff[23])), xgrad.at<float>(pt0 + diff[23]));
	float direction24 = atan2(abs(ygrad.at<float>(pt0 + diff[24])), xgrad.at<float>(pt0 + diff[24]));


	A[0] = direction0; A[1] = direction1; A[2] = direction2; A[3] = direction3; A[4] = direction4;
	A[5] = direction5; A[6] = direction6; A[7] = direction7; A[8] = direction8; A[9] = direction9;
	A[10] = direction10; A[11] = direction11; A[12] = direction12; A[13] = direction13; A[14] = direction14;
	A[15] = direction15; A[16] = direction16; A[17] = direction17; A[18] = direction18; A[19] = direction19;
	A[20] = direction20; A[21] = direction21; A[22] = direction22; A[23] = direction23; A[24] = direction24;
	QuickSort(A, 0, 25);
}


void SpreadGradient(Mat&J, Mat&J_origin, vector<Point2f>diff, imageInfo srcImgInfo)
{
	Point2f pt;
	for (int i = 0; i < srcImgInfo.featurePoints.size(); i++)
	{
		pt = Point2f(srcImgInfo.featurePoints[i].x, srcImgInfo.featurePoints[i].y);
		J_origin.at<uchar>(pt) |= srcImgInfo.oris[i];
		if (pt.x > 1 && pt.x < J.cols - 2 && pt.y > 1 && pt.y < J.rows - 2)
		{
			for (int k = 0; k < diff.size(); k++)
			{
				J.at<uchar>(pt + diff[k]) |= srcImgInfo.oris[i];
			}
		}
	}
}


/*建立查找表*/
float(*EstablishLookupTable(float(&Map)[256 - 1][8]))[8]
{
	for (int i = 1; i <= 256 - 1; i++)
	{
		uchar oriJ = (uchar)i;
		for (int j = 0; j < 8; j++)
		{
			if ((int)((int)(oriJ&(int)(pow(2, j))) > 0) == 1)Map[i - 1][j] = 1;
			else
			{
				int jH = 1, jL = 1;
				while ((int)((int)(oriJ&(int)(pow(2, j + jH))) > 0) == 0)
				{
					jH++;
					if (j + jH > 8 - 1)
					{
						if ((int)((int)(oriJ&(int)(pow(2, 8 - 1))) > 0) == 0)
						{
							jH = 8;
						}
						break;
					}
				}
				while ((int)((int)(oriJ&(int)(pow(2, j - jL))) > 0) == 0)
				{
					jL++;
					if (j - jL < 0)
					{
						if ((int)((int)(oriJ&(int)(pow(2, 0))) > 0) == 0)
						{
							jL = 8;
						}
						break;
					}
				}
				int jC = jH < jL ? jH : jL;
				Map[i - 1][j] = abs(cos(jC * PI / 8));
			}
		}
	}

	return Map;
}


/*查表法计算最大余弦绝对值
oriT: 模板图像有效梯度点的方向，总计八个
oriJ: 图像J上梯度方向，由于经过gradient spread，可能是多个方向的组合bitCode*/
float calMaxAbsCosValue(uchar oriT, uchar oriJ, float(*tab)[8])
{
	int J_row = (int)(oriJ)-1;
	int J_col = 0;
	while ((int)(oriT & (int)(pow(2, J_col))) == 0)
	{
		J_col++;
	}
	float result = tab[J_row][J_col];
	return result;
}



vector<FastMatchTrans>FirstTransNet(vector<Translation> &transNet, vector<Rotation> &rotateNet,
	Mat &resizeSrcImage, Mat &maskImgInfo,Mat &J, float(*tab)[8], float resizeFactor,
	float maskImgCol, float maskImgRow, int srcFeatureNum,float sampleNum)
{
	vector<float>valueList;
	vector<FastMatchTrans>tempTransNet, resultTransNet;
	int k = 0;
	for (int col = 0; col < resizeSrcImage.cols; col += sampleNum)
	{
		for (int row = 0; row < resizeSrcImage.rows; row += sampleNum)
		{
			int index = (resizeSrcImage.rows + 1)*row + col;
			if (index>transNet.size()-1)
			{
				continue;
			}

			float sx = transNet[index].x;
			float sy = transNet[index].y;
			Point2f pt = Point2f(sx, sy);

			for (int thetaIndex = 0; thetaIndex < rotateNet.size(); thetaIndex += 20)
			{
				float stheta = rotateNet[thetaIndex].theta;
				float tempSumValue = 0;
				for (int i = 0; i < maskImgInfo.cols; i++)
				{
					float mx = maskImgInfo.at<Vec4f>(0, i)[0] - maskImgCol / 2;
					float my = maskImgInfo.at<Vec4f>(0, i)[1] - maskImgRow / 2;
					float rsx = cos(stheta) * mx - sin(stheta) * my;
					float rsy = sin(stheta) * mx + cos(stheta) * my;
					float x = sx + rsx;
					float y = sy + rsy;
					Point2f pt2(x, y);//图像J上对应的点

					float orientation = maskImgInfo.at<Vec4f>(0, i)[2];
					uchar oriT = CalOrientationCode(orientation);//模板T上一点的方向

					//计算当前点在输入图像上对应的点的领域内的绝对余弦最大值
					if (pt2.x < 0 || pt2.y < 0 || pt2.x > J.cols - 1 || pt2.y > J.rows - 1)	continue;
					uchar oriJ = J.at<uchar>(pt2);
					if ((int)(oriJ) == 0)continue;
					tempSumValue += calMaxAbsCosValue(oriT, oriJ, tab);
				}
				if (tempSumValue > maskImgInfo.cols * 0.1)
				{
					valueList.push_back(tempSumValue);
					FastMatchTrans temp;
					temp.theta = rotateNet[thetaIndex].theta;
					temp.x = transNet[index].x;
					temp.y = transNet[index].y;
					tempTransNet.push_back(temp);
					k++;
				}
			}
		}
	}

	int resultTransNetSize = 5 > tempTransNet.size() ? tempTransNet.size() : 5;
	for (int i = 0; i < resultTransNetSize; i++)
	{
		vector<float>::iterator biggest = max_element(valueList.begin(), valueList.end());
		int maxPosition = distance(valueList.begin(), biggest);
		resultTransNet.push_back(tempTransNet[maxPosition]);
		tempTransNet.erase(tempTransNet.begin() + maxPosition);
		valueList.erase(valueList.begin() + maxPosition);
	}

	return resultTransNet;
}


vector<FastMatchTrans>ExpandTransNet(vector<FastMatchTrans>& firstTrans)
{
	FastMatchTrans temp;
	vector<FastMatchTrans> resultTransNet = firstTrans;
	for (int i = 0; i < firstTrans.size(); i ++)
	{
		for (float x = -8; x < 8; x+=2)
		{
			for (float y = -8; y < 8; y+=2)
			{
				for (float theta = -5*PI / 180; theta < 5*PI / 180; theta += 0.5*PI / 180)
				{
					temp.theta = theta+firstTrans[i].theta;
					temp.x = x + firstTrans[i].x * 2;
					temp.y = y + firstTrans[i].y * 2;
					resultTransNet.push_back(temp);
				}
			}
		}
	}

	return resultTransNet;
}


FastMatchTrans GetBestTrans(vector<FastMatchTrans>& expandTrans, Mat&resizeSrcImage, Mat&maskImgInfo,
	Mat &J, float(*tab)[8], float resizeFactor, float maskImgCol, float maskImgRow, float sampleNum)
{
	FastMatchTrans result;
	float sumSimilarity = 0.0f;
	for (int i = 0; i < expandTrans.size(); i++)
	{
		float sx = expandTrans[i].x;
		float sy = expandTrans[i].y;
		float stheta = expandTrans[i].theta;

		Point2f pt = Point2f(sx, sy);
		float tempSumValue = 0;
		for (int i = 0; i < maskImgInfo.cols; i++)
		{
			float mx = maskImgInfo.at<Vec4f>(0, i)[0] - maskImgCol / 2;
			float my = maskImgInfo.at<Vec4f>(0, i)[1] - maskImgRow / 2;
			float rsx = cos(stheta)*mx - sin(stheta)*my;
			float rsy = sin(stheta)*mx + cos(stheta)*my;
			float x = sx + (float)(rsx / pow(2, resizeFactor));
			float y = sy + (float)(rsy / pow(2, resizeFactor));
			Point2f pt2(x, y);//图像J上对应的点

			float orientation = maskImgInfo.at<Vec4f>(0, i)[2];
			uchar oriT = CalOrientationCode(orientation);//模板T上一点的方向

			//计算当前点在输入图像上对应的点的领域内的绝对余弦最大值
			if (pt2.x < 0 || pt2.y < 0 || pt2.x > J.cols - 1 || pt2.y > J.rows - 1)
			{
				continue;
			}
			uchar oriJ = J.at<uchar>(pt2);
			if ((int)(oriJ) == 0)continue;
			tempSumValue += calMaxAbsCosValue(oriT, oriJ, tab);
		}
		if (tempSumValue > sumSimilarity)
		{
			sumSimilarity = tempSumValue;
			result.theta = stheta;
			result.x = sx;
			result.y = sy;
		}
	}

	return result;
}


vector<FastMatchTrans> FastMatch(Mat&resizeSrcImage, int srcFeatureNum, Mat&maskImgInfo, Mat &J, float(*tab)[8],
	float resizeFactor, float maskImgCol, float maskImgRow, vector<Translation>&TransNet, vector<Rotation>&RotateNet)
{
	float sampleNum = 4.0f;
	vector<FastMatchTrans> firstTransNet = FirstTransNet(TransNet, RotateNet, resizeSrcImage, maskImgInfo, J, tab, resizeFactor, maskImgCol, maskImgRow, srcFeatureNum, sampleNum);
	return firstTransNet;
}

//
//TransParm2 GetBestTransParm(Mat &resizeSrcImageL, Mat &resizeSrcImageR, float resizeFactor,
//	Mat &maskImgLInfo, Mat &maskImgRInfo, float maskImgLCol, float maskImgLRow, float maskImgRCol,
//	float maskImgRRow, Mat &JL, Mat &JR, float(*tab)[8], imageInfo&srcImgLInfo, imageInfo&srcImgRInfo)
//{
//	/*step 1:建立变换网络*/
//	vector<Translation> TransNet;
//	vector<Rotation>RotateNet;
//	ConstructTransNet(TransNet, resizeSrcImageL.cols, resizeSrcImageL.rows);
//	float minAngle = -10.0 * PI / 180;
//	float maxAngle = 10 * PI / 180;
//	float step = 0.1 * PI / 180;
//	ConstructRotateNet(RotateNet,minAngle,maxAngle,step);
//	vector<FastMatchTrans> resultL = FastMatch(resizeSrcImageL, srcImgLInfo.featurePoints.size(),
//		maskImgLInfo, JL, tab, resizeFactor, maskImgLCol, maskImgLRow, TransNet, RotateNet);
//	vector<FastMatchTrans> resultR = FastMatch(resizeSrcImageR, srcImgRInfo.featurePoints.size(),
//		maskImgRInfo, JR, tab, resizeFactor, maskImgRCol, maskImgRRow, TransNet, RotateNet);
//	TransParm2 result;
//	result.bestLocationL = Point2f(resultL.x, resultL.y);
//	result.bestLocationR = Point2f(resultR.x, resultR.y);
//	result.bestThetaL = resultL.theta;
//	result.bestThetaR = resultR.theta;
//	return result;
//}


TransParm2 GetBestTransParm(TransParm2 coarseLocation, Mat &resizeSrcImageL, Mat &resizeSrcImageR, float resizeFactor,
	Mat &maskImgLInfo, Mat &maskImgRInfo, float maskImgLCol, float maskImgLRow, float maskImgRCol, float maskImgRRow,
	Mat &JL, Mat &JR, float(*tab)[8], imageInfo&srcImgLInfo, imageInfo&srcImgRInfo)
{
	float  sumSimilarity = 0;
	Point2f bestLocationL, bestLocationR;
	float bestThetaL, bestThetaR;
	for (float sxl = -10 + coarseLocation.bestLocationL.x; sxl < 10 + coarseLocation.bestLocationL.x; sxl++)//在输入图像上搜索
	{
		for (float syl = -10 + coarseLocation.bestLocationL.y; syl < 10 + coarseLocation.bestLocationL.y; syl++)
		{
			for (float sthetal = -0.5 * PI / 180; sthetal < 0.5 * PI / 180; sthetal += 0.1 * PI / 180)
			{
				Point2f pt = Point2f(sxl, syl);
				float tempSumValue = 0;
				for (int i = 0; i < maskImgLInfo.cols; i++)
				{
					float mx = maskImgLInfo.at<Vec4f>(0, i)[0] - maskImgLCol / 2;
					float my = maskImgLInfo.at<Vec4f>(0, i)[1] - maskImgLRow / 2;
					float rsxl = cos(sthetal)*mx - sin(sthetal)*my;
					float rsyl = sin(sthetal)*mx + cos(sthetal)*my;
					float x = sxl + (float)(rsxl / pow(2, resizeFactor));
					float y = syl + (float)(rsyl / pow(2, resizeFactor));
					Point2f pt2(x, y);//图像J上对应的点

					float orientation = maskImgLInfo.at<Vec4f>(0, i)[2];
					uchar oriT = CalOrientationCode(orientation);//模板T上一点的方向

					//计算当前点在输入图像上对应的点的领域内的绝对余弦最大值
					if (pt2.x < 0 || pt2.y < 0 || pt2.x > JL.cols - 1 || pt2.y > JL.rows - 1)
					{
						continue;
					}
					uchar oriJ = JL.at<uchar>(pt2);
					if ((int)(oriJ) == 0)continue;
					tempSumValue += calMaxAbsCosValue(oriT, oriJ, tab);
				}
				if (tempSumValue > sumSimilarity)
				{
					sumSimilarity = tempSumValue;
					bestLocationL = pt;
					bestThetaL = sthetal;
				}
			}
		}
	}

	sumSimilarity = 0.f;
	for (float sxr = -10 + coarseLocation.bestLocationR.x; sxr < 10 + coarseLocation.bestLocationR.x; sxr++)//在输入图像上搜索
	{
		for (float syr = -10 + coarseLocation.bestLocationR.y; syr < 10 + coarseLocation.bestLocationR.y; syr++)
		{
			for (float sthetar = -0.5 * PI / 180; sthetar < 0.5 * PI / 180; sthetar += 0.1 * PI / 180)
			{
				Point2f pt = Point2f(sxr, syr);
				float tempSumValue = 0;
				for (int i = 0; i < maskImgRInfo.cols; i++)
				{
					float mx = maskImgRInfo.at<Vec4f>(0, i)[0] - maskImgRCol / 2;
					float my = maskImgRInfo.at<Vec4f>(0, i)[1] - maskImgRRow / 2;
					float rsxr = cos(sthetar)*mx - sin(sthetar)*my;
					float rsyr = sin(sthetar)*mx + cos(sthetar)*my;
					float x = sxr + (float)(rsxr / pow(2, resizeFactor));
					float y = syr + (float)(rsyr / pow(2, resizeFactor));
					Point2f pt2(x, y);//图像J上对应的点

					float orientation = maskImgRInfo.at<Vec4f>(0, i)[2];
					uchar oriT = CalOrientationCode(orientation);//模板T上一点的方向

					//计算当前点在输入图像上对应的点的领域内的绝对余弦最大值
					if (pt2.x < 0 || pt2.y < 0 || pt2.x > JR.cols - 1 || pt2.y > JR.rows - 1)
					{
						continue;
					}
					uchar oriJ = JR.at<uchar>(pt2);
					if ((int)(oriJ) == 0)continue;
					tempSumValue += calMaxAbsCosValue(oriT, oriJ, tab);
				}
				if (tempSumValue > sumSimilarity)
				{
					sumSimilarity = tempSumValue;
					bestLocationR = pt;
					bestThetaR = sthetar;
				}
			}
		}
	}

	TransParm2 result;
	result.bestLocationL = bestLocationL;
	result.bestLocationR = bestLocationR;
	result.bestThetaL = bestThetaL;
	result.bestThetaR = bestThetaR;
	return result;
}



void ConstructTransNet(vector<Translation>&net, int searchCol, int searchRow)
{
	Translation temp;
	for (float col = 0; col < searchCol; col++)
	{
		for (float row = 0; row < searchRow; row++)
		{
			temp.x = col;
			temp.y = row;
			net.push_back(temp);
		}
	}
}



void ConstructRotateNet(vector<Rotation>&net,float minAngle,float maxAngle,float step)
{
	Rotation temp;
	for (float theta = minAngle; theta <= maxAngle; theta += step)
	{
		temp.theta = theta;
		net.push_back(temp);
	}
}


vector<FastMatchTrans> CoarseShapeMatch(Mat &srcImg,int resizeFactor,
	Mat &maskImgInfo, float maskImgCol, float maskImgRow, float(*tab)[8])
{
	Mat resizeSrcImage;
	resize(srcImg, resizeSrcImage, Size(srcImg.cols / pow(2, resizeFactor),
		srcImg.rows / pow(2, resizeFactor)));

	//process the srcImage's feature information.
	int maskFeatureNum = maskImgInfo.cols;
	int srcFeatureNum = 100;
	imageInfo srcImgInfo;
	srcImgInfo = ProcessImage(resizeSrcImage, srcFeatureNum);

	/*quantify domain orientation.*/
	vector<Point2f>diff;
	for (float col = -2; col <= 2; col++)
	{
		for (float row = -2; row <= 2; row++)
		{
			Point2f pt(col, row);
			diff.push_back(pt);
		}
	}

	for (int i = 0; i < srcImgInfo.featurePoints.size(); i++)
	{
		Point2f pt(srcImgInfo.featurePoints[i].x, srcImgInfo.featurePoints[i].y);
		float directionList[25];
		//输出当前点的邻域内的梯度方向排序列表，取中位数作为当前点的梯度方向
		GetSortedDirListInRegion(directionList, srcImgInfo.xgrad, srcImgInfo.ygrad, pt, diff);
		uchar ori = CalOrientationCode(directionList[int(diff.size() / 2)]);
		srcImgInfo.oris.push_back(ori);
	}

	//spread the gradient to J
	Mat J_origin = Mat::zeros(resizeSrcImage.size(), CV_8UC1);
	Mat J = Mat::zeros(resizeSrcImage.size(), CV_8UC1);
	SpreadGradient(J, J_origin, diff, srcImgInfo);

	vector<Translation> TransNet;
	vector<Rotation>RotateNet;
	ConstructTransNet(TransNet, resizeSrcImage.cols, resizeSrcImage.rows);
	float minAngle = -10.0 * PI / 180, maxAngle = 10 * PI / 180;
	float step = 0.1 * PI / 180;
	ConstructRotateNet(RotateNet, minAngle, maxAngle, step);

	vector<FastMatchTrans> result = FastMatch(resizeSrcImage, srcImgInfo.featurePoints.size(),
		maskImgInfo, J, tab, resizeFactor, maskImgCol, maskImgRow, TransNet, RotateNet);

	return result;
}



vector<FastMatchTrans> RefineShapeMatch(vector<FastMatchTrans> expandTransNet,
	int outputListNum, Mat &srcImg, int resizeFactor, Mat &maskImgInfo,
	float maskImgCol, float maskImgRow, float(*tab)[8])
{
	Mat resizeSrcImage;
	resize(srcImg, resizeSrcImage,
		Size(srcImg.cols / pow(2, resizeFactor), srcImg.rows / pow(2, resizeFactor)));

	//process the srcImage's feature information
	int maskFeatureNum = maskImgInfo.cols;
	//int srcFeatureNum = maskFeatureNum * (int)(resizeSrcImage.cols / maskFeatureNum);
	imageInfo srcImgInfo = ProcessImage(resizeSrcImage, 99999);

	/*quantify domain orientation*/
	vector<Point2f>diff;
	for (float col = -2; col <= 2; col++)
	{
		for (float row = -2; row <= 2; row++)
		{
			Point2f pt(col, row);
			diff.push_back(pt);
		}
	}

	for (int i = 0; i < srcImgInfo.featurePoints.size(); i++)
	{
		Point2f pt(srcImgInfo.featurePoints[i].x, srcImgInfo.featurePoints[i].y);
		float directionList[25];
		//输出当前点的邻域内的梯度方向排序列表，取中位数作为当前点的梯度方向
		GetSortedDirListInRegion(directionList, srcImgInfo.xgrad, srcImgInfo.ygrad, pt, diff);
		uchar ori = CalOrientationCode(directionList[int(diff.size() / 2)]);
		srcImgInfo.oris.push_back(ori);
	}

	//spread the gradient to J
	Mat J_origin = Mat::zeros(resizeSrcImage.size(), CV_8UC1);
	Mat J = Mat::zeros(resizeSrcImage.size(), CV_8UC1);
	SpreadGradient(J, J_origin, diff, srcImgInfo);
	//Total time is 1.8ms.

	vector<float>valueList;
	vector<FastMatchTrans>tempTransNet, resultTransNet;
	int k = 0;
	for (int i = 0; i < expandTransNet.size(); i++)
	{
		float sx = expandTransNet[i].x;
		float sy = expandTransNet[i].y;
		float stheta = expandTransNet[i].theta;
		Point2f pt = Point2f(sx, sy);

		float tempSumValue = 0;
		for (int i = 0; i < maskImgInfo.cols; i++)
		{
			float mx = maskImgInfo.at<Vec4f>(0, i)[0] - maskImgCol / 2;
			float my = maskImgInfo.at<Vec4f>(0, i)[1] - maskImgRow / 2;
			float rsx = cos(stheta) * mx - sin(stheta) * my;
			float rsy = sin(stheta) * mx + cos(stheta) * my;
			float x = sx + rsx;
			float y = sy + rsy;
			Point2f pt2(x, y);//图像J上对应的点

			float orientation = maskImgInfo.at<Vec4f>(0, i)[2];
			uchar oriT = CalOrientationCode(orientation);//模板T上一点的方向

			//计算当前点在输入图像上对应的点的领域内的绝对余弦最大值
			if (pt2.x < 0 || pt2.y < 0 || pt2.x > J_origin.cols - 1 || pt2.y > J_origin.rows - 1)	continue;
			uchar oriJ = J_origin.at<uchar>(pt2);
			if ((int)(oriJ) == 0)continue;
			tempSumValue += calMaxAbsCosValue(oriT, oriJ, tab);
		}
		if (tempSumValue > 0/*maskImgInfo.cols*0.5*/)
		{
			valueList.push_back(tempSumValue);
			tempTransNet.push_back(expandTransNet[i]);
			k++;
		}
	}
	int resultTransNetSize = outputListNum > tempTransNet.size() ? tempTransNet.size() : outputListNum;
	for (int i = 0; i < resultTransNetSize; i++)
	{
		vector<float>::iterator biggest = max_element(valueList.begin(), valueList.end());
		int maxPosition = distance(valueList.begin(), biggest);
		resultTransNet.push_back(tempTransNet[maxPosition]);
		tempTransNet.erase(tempTransNet.begin() + maxPosition);
		valueList.erase(valueList.begin() + maxPosition);
	}

	return resultTransNet;
}
