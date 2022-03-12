<<<<<<< HEAD
#include<opencv2\imgproc\imgproc_c.h>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string.h>
#include<math.h>
#include<vector>
#define    CLOCKS_PER_SEC      ((clock_t)1000)
#define    BLACK     0
#define    WHITE     255
using namespace std;
using namespace cv;

static string testImgPath = "./签名/*";//图片路径

int  medianValue(int* pixels, int size)//取中值
{
	sort(pixels, pixels + size);//升序排序
	return pixels[size / 2];//返回中值
}
void toGrey(Mat& Img)//转化为灰度图像
{
	int row = Img.rows;//复制行数
	int col = Img.cols;//列数
	Mat greyImg;//定义灰度图
	greyImg.create(row, col, CV_8UC1);//初始化大小
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
		{
			//得到三个通道的像素值
			int B = Img.at<Vec3b>(i, j)[0];
			int G = Img.at<Vec3b>(i, j)[1];
			int R = Img.at<Vec3b>(i, j)[2];
			//利用灰度化公式将彩色图像三个通道的像素值转化为灰度图像单通道的像素值
			greyImg.at<uchar>(i, j) = static_cast<uchar>(0.114 * B + 0.587 * G + 0.299 * R);
		}
	Img = greyImg;
}

void quickSort(vector<int>& a, int low, int high)
{
	if (low < high)  //判断是否满足排序条件，递归的终止条件
	{
		int i = low, j = high;
		int x = a[low];
		while (i < j)
		{
			while (i < j && a[j] >= x) j--;
			if (i < j) a[i++] = a[j];
			while (i < j && a[i] <= x) i++;
			if (i < j) a[j--] = a[i];
		}
		a[i] = x;
		quickSort(a, low, i - 1);
		quickSort(a, i + 1, high);
	}
}

int bubbleMedian(vector<int>& a)//得到中值，快排
{
	int median = 0;//找到中值
	quickSort(a, 0, 8);
	median = a[4];
	return median;
}

void denoising(Mat gray1, Mat& grayImg)//第三步 去噪处理（中值滤波）,实测比均值滤波效果要好一点
{
	grayImg = Mat(gray1.rows, gray1.cols, CV_8UC1);//新建一个一样大小的图片

	vector<int>temp(9); //定义九个方向（相对坐标）
	int dx[9] = { 1,-1,1,-1,-1,0,1,0,0 };
	int dy[9] = { 1,-1,-1,1,0,1,0,-1,0 };

	for (int i = 1; i < gray1.rows - 1; i++)
	{
		for (int j = 1; j < gray1.cols - 1; j++)
		{
			//边缘部分不做处理
			{
				for (int k = 0; k < 9; k++)
				{
					temp[k] = gray1.at<uchar>(i + dx[k], j + dy[k]);//得到绝对坐标
				}
				grayImg.at<uchar>(i, j) = bubbleMedian(temp);
			}
		}
	}
}

int getThreshold(Mat& Img)//大津阈值法
{
	//1.绘制灰度直方图
	int fig[256] = { 0 };//直方图数组
	for (int i = 0; i < Img.rows; i++)
		for (int j = 0; j < Img.cols; j++)
			fig[Img.at<uchar>(i, j)]++;

	//2.计算阈值
	int Threshold;//阈值
	double var = 0;//类间方差
	int sum = Img.rows * Img.rows;
	for (int i = 0; i < 255; i++)//枚举阈值
	{
		double firstAVG = 0, firstRate = 0, backAVG = 0, backRate = 0;
		//前景色平均灰度，前景色像素占比，背景色平均灰度，背景色像素占比
		for (int j = 0; j < 256; j++)
			if (j <= i)firstAVG += fig[j] * j, firstRate += fig[j];//统计前景色的总灰度值和总像素数
			else  backAVG += fig[j] * j, backRate += fig[j];//统计背景色的总灰度值和总像素数
		firstAVG /= firstRate, firstRate /= sum;//计算前景色的平均灰度值和像素占比
		backAVG /= backRate, backRate /= sum;//计算背景色的平均灰度值和像素占比
		double var1 = firstRate * backRate * (firstAVG - backAVG) * (firstAVG - backAVG);
		if (var1 > var)var = var1, Threshold = i;
	}
	return Threshold;
}


void FLoodFill(Mat& Img, int x0, int y0)//八邻域填二值图充边界黑色部分为白色
{
	queue<pair<int, int>> q;
	int move[8][2] = { {-1,-1},{-1,0},{-1,1},{0,1},{0,-1},{1,-1},{1,0},{1,1} };//九个方向
	Img.at<uchar>(x0, y0) = 0;
	q.push({ x0,y0 });
	while (q.size())
	{
		auto t = q.front();
		q.pop();
		for (int i = 0; i < 8; i++)
		{
			int x = t.first + move[i][0];
			int y = t.second + move[i][1];
			if (x >= 0 && x < Img.rows && y >= 0 &&
				y < Img.cols && Img.at<uchar>(x, y) == WHITE)//白色
				q.push({ x,y }), Img.at<uchar>(x, y) = BLACK;//黑色
		}
	}
}
void toBinaryGraph(Mat& Img, bool Invert = 1, bool FillEdege = 1)//将灰度图二值化，参数列表：二值化的图像，是否反相，是否填充边缘
{
	int Threshold = getThreshold(Img);// 阈值
	int front = WHITE, back = BLACK;//默认前景色为白色，底色为黑色
	if (!Invert)front = BLACK, back = WHITE;
	for (int i = 0; i < Img.rows; i++)
		for (int j = 0; j < Img.cols; j++)
			if (Img.at<uchar>(i, j) > Threshold)Img.at<uchar>(i, j) = back;//大于阈值设为黑色
			else Img.at<uchar>(i, j) = front;//小于阈值设为白色
	if (!FillEdege)return;//不需要填充边缘，则返回
	//防止IBSN受损，对边缘进行填充
	for (int i = Img.rows / 2; i < Img.rows; i++)
	{
		if (Img.at<uchar>(i, 0) == WHITE)FLoodFill(Img, i, 0);
		if (Img.at<uchar>(i, Img.cols - 1) == WHITE)FLoodFill(Img, i, Img.cols - 1);
	}
	for (int i = 0; i < Img.cols; i++)
	{
		if (Img.at<uchar>(0, i) == WHITE)FLoodFill(Img, 0, i);
		if (Img.at<uchar>(Img.rows - 1, i) == WHITE)FLoodFill(Img, Img.rows - 1, i);
	}

}
void revolve(Mat& Img, Mat& revolveMatrix, Size& NewSize)//旋转图片到合适的位置
{
	Mat Img_x;
	Sobel(Img, Img_x, -1, 0, 1, 5);
	vector<Vec2f>Lines;
	HoughLines(Img_x, Lines, 1, CV_PI / 180, 180);
	double angle = 0;
	int cnt = 0;
	for (auto i : Lines)
		if (i[1] > CV_PI * 17.0 / 36 && i[1] < CV_PI * 5.0 / 9 && i[0] < Img.rows / 3)
			//过滤直线：倾角在（85°,100°）
			angle += i[1], cnt++;
	angle /= cnt;

	revolveMatrix = getRotationMatrix2D(Point(Img.rows / 2, Img.cols / 2), 180 * angle / CV_PI - 90, 1);
	//逆时针旋转，获得旋转矩阵
	//根据旋转矩阵，计算旋转后图像的大小，为了防止边界的ISBN的数字丢失
	double cos = abs(revolveMatrix.at<double>(0, 0));
	double sin = abs(revolveMatrix.at<double>(0, 1));
	int nw = cos * Img.cols + sin * Img.rows;
	int nh = sin * Img.cols + cos * Img.rows;
	revolveMatrix.at<double>(0, 2) += (nw / 2 - Img.cols / 2);
	revolveMatrix.at<double>(1, 2) += (nh / 2 - Img.rows / 2);
	NewSize = { nw,nh };
}

int rowSum(Mat& Img, int rowi)//一行像素灰度值之和
{
	int sum = 0;
	for (int i = 0; i < Img.cols; i++)
		if (Img.at<uchar>(rowi, i) == 255)sum++;
	return sum;
}
int colSum(Mat& Img, int coli)//一列像素灰度值之和
{
	int sum = 0;
	for (int i = 0; i < Img.rows; i++)
		if (Img.at<uchar>(i, coli) == 255)sum++;
	return sum;
}
int allSum(Mat& Img)
{
	int sum = 0;
	for (int i = 0; i < Img.rows; i++)
		for (int j = 0; j < Img.cols; j++)
			if (Img.at<uchar>(i, j) == 255)sum++;
	return sum;
}

void getRow(Mat& Img)//获得ISBN的所在行
{
	double width = 1500;
	double height = width * Img.rows / Img.cols;
	resize(Img, Img, Size(width, height));
	toGrey(Img);
	denoising(Img, Img);
	toBinaryGraph(Img);
	Size newsize;
	Mat m;
	revolve(Img, m, newsize);
	warpAffine(Img, Img, m, newsize, INTER_LINEAR, 0);
	//找ISBN所在行的上下界
	int rowi = 0, Uper = 0, Lower = 0, Height = 0;
	while (rowi < Img.rows / 2)
	{
		while (rowi < Img.rows / 2 && rowSum(Img, rowi) < 10)rowi++;
		int upper = rowi;
		while (rowi < Img.rows / 2 && rowSum(Img, rowi) >= 10)rowi++;
		int lower = rowi;
		if (rowi < Img.rows / 2 && lower - upper > Height)
			Uper = upper, Lower = lower, Height = lower - upper;
	}
	if (Uper < Lower)
		Img = Mat(Img, Range(Uper, Lower), Range(0, Img.cols));
}


void getMinRec(Mat& Img)//获得最小矩阵
{
	int uper, lower;
	int rowi = 0;
	while (!rowSum(Img, rowi))rowi++;
	uper = rowi;
	rowi = Img.rows - 1;
	while (!rowSum(Img, rowi))rowi--;
	lower = rowi;
	if (uper < lower)
		Img = Mat(Img, Range(uper, lower), Range(0, Img.cols));

	int lefter, righter;
	int coli = 0;
	while (!colSum(Img, coli))coli++;
	lefter = coli;
	coli = Img.cols - 1;
	while (!colSum(Img, coli))coli--;
	righter = coli;
	if (lefter < righter)
		Img = Mat(Img, Range(0, Img.rows), Range(lefter, righter));

}

int main()
{
	int rtNums = 0, accNums = 0, sunNums = 0;//分别代表：正确的数量，被准确识别的字符的数量，要识别的字符的总和

	//1.读取 ISBN 图片

	vector<String> testImgFN;
	glob(testImgPath, testImgFN, false);//将所有 符合条件的图片 放入 testImgFN中
	int testImgNums = testImgFN.size();//得到图片总数

	clock_t start = clock();
	for (int i = 0; i < testImgNums; i++)
	{
		Mat Img = imread(testImgFN[i]);
		getRow(Img);
		getMinRec(Img);
		imshow("二值图像", Img);
		waitKey(6000);
	}
	clock_t end = clock();
	double timeSpend = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "共生成了：" << testImgNums << "张图片。 " << timeSpend / testImgNums << " Sec/prePic" << endl;

}
=======
#include<opencv2\imgproc\imgproc_c.h>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string.h>
#include<math.h>
#include<vector>
#define    CLOCKS_PER_SEC      ((clock_t)1000)
#define    BLACK     0
#define    WHITE     255
using namespace std;
using namespace cv;

static string testImgPath = "./签名/*";//图片路径

int  medianValue(int* pixels, int size)//取中值
{
	sort(pixels, pixels + size);//升序排序
	return pixels[size / 2];//返回中值
}
void toGrey(Mat& Img)//转化为灰度图像
{
	int row = Img.rows;//复制行数
	int col = Img.cols;//列数
	Mat greyImg;//定义灰度图
	greyImg.create(row, col, CV_8UC1);//初始化大小
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
		{
			//得到三个通道的像素值
			int B = Img.at<Vec3b>(i, j)[0];
			int G = Img.at<Vec3b>(i, j)[1];
			int R = Img.at<Vec3b>(i, j)[2];
			//利用灰度化公式将彩色图像三个通道的像素值转化为灰度图像单通道的像素值
			greyImg.at<uchar>(i, j) = static_cast<uchar>(0.114 * B + 0.587 * G + 0.299 * R);
		}
	Img = greyImg;
}

void quickSort(vector<int>& a, int low, int high)
{
	if (low < high)  //判断是否满足排序条件，递归的终止条件
	{
		int i = low, j = high;
		int x = a[low];
		while (i < j)
		{
			while (i < j && a[j] >= x) j--;
			if (i < j) a[i++] = a[j];
			while (i < j && a[i] <= x) i++;
			if (i < j) a[j--] = a[i];
		}
		a[i] = x;
		quickSort(a, low, i - 1);
		quickSort(a, i + 1, high);
	}
}

int bubbleMedian(vector<int>& a)//得到中值，快排
{
	int median = 0;//找到中值
	quickSort(a, 0, 8);
	median = a[4];
	return median;
}

void denoising(Mat gray1, Mat& grayImg)//第三步 去噪处理（中值滤波）,实测比均值滤波效果要好一点
{
	grayImg = Mat(gray1.rows, gray1.cols, CV_8UC1);//新建一个一样大小的图片

	vector<int>temp(9); //定义九个方向（相对坐标）
	int dx[9] = { 1,-1,1,-1,-1,0,1,0,0 };
	int dy[9] = { 1,-1,-1,1,0,1,0,-1,0 };

	for (int i = 1; i < gray1.rows - 1; i++)
	{
		for (int j = 1; j < gray1.cols - 1; j++)
		{
			//边缘部分不做处理
			{
				for (int k = 0; k < 9; k++)
				{
					temp[k] = gray1.at<uchar>(i + dx[k], j + dy[k]);//得到绝对坐标
				}
				grayImg.at<uchar>(i, j) = bubbleMedian(temp);
			}
		}
	}
}

int getThreshold(Mat& Img)//大津阈值法
{
	//1.绘制灰度直方图
	int fig[256] = { 0 };//直方图数组
	for (int i = 0; i < Img.rows; i++)
		for (int j = 0; j < Img.cols; j++)
			fig[Img.at<uchar>(i, j)]++;

	//2.计算阈值
	int Threshold;//阈值
	double var = 0;//类间方差
	int sum = Img.rows * Img.rows;
	for (int i = 0; i < 255; i++)//枚举阈值
	{
		double firstAVG = 0, firstRate = 0, backAVG = 0, backRate = 0;
		//前景色平均灰度，前景色像素占比，背景色平均灰度，背景色像素占比
		for (int j = 0; j < 256; j++)
			if (j <= i)firstAVG += fig[j] * j, firstRate += fig[j];//统计前景色的总灰度值和总像素数
			else  backAVG += fig[j] * j, backRate += fig[j];//统计背景色的总灰度值和总像素数
		firstAVG /= firstRate, firstRate /= sum;//计算前景色的平均灰度值和像素占比
		backAVG /= backRate, backRate /= sum;//计算背景色的平均灰度值和像素占比
		double var1 = firstRate * backRate * (firstAVG - backAVG) * (firstAVG - backAVG);
		if (var1 > var)var = var1, Threshold = i;
	}
	return Threshold;
}


void FLoodFill(Mat& Img, int x0, int y0)//八邻域填二值图充边界黑色部分为白色
{
	queue<pair<int, int>> q;
	int move[8][2] = { {-1,-1},{-1,0},{-1,1},{0,1},{0,-1},{1,-1},{1,0},{1,1} };//九个方向
	Img.at<uchar>(x0, y0) = 0;
	q.push({ x0,y0 });
	while (q.size())
	{
		auto t = q.front();
		q.pop();
		for (int i = 0; i < 8; i++)
		{
			int x = t.first + move[i][0];
			int y = t.second + move[i][1];
			if (x >= 0 && x < Img.rows && y >= 0 &&
				y < Img.cols && Img.at<uchar>(x, y) == WHITE)//白色
				q.push({ x,y }), Img.at<uchar>(x, y) = BLACK;//黑色
		}
	}
}
void toBinaryGraph(Mat& Img, bool Invert = 1, bool FillEdege = 1)//将灰度图二值化，参数列表：二值化的图像，是否反相，是否填充边缘
{
	int Threshold = getThreshold(Img);// 阈值
	int front = WHITE, back = BLACK;//默认前景色为白色，底色为黑色
	if (!Invert)front = BLACK, back = WHITE;
	for (int i = 0; i < Img.rows; i++)
		for (int j = 0; j < Img.cols; j++)
			if (Img.at<uchar>(i, j) > Threshold)Img.at<uchar>(i, j) = back;//大于阈值设为黑色
			else Img.at<uchar>(i, j) = front;//小于阈值设为白色
	if (!FillEdege)return;//不需要填充边缘，则返回
	//防止IBSN受损，对边缘进行填充
	for (int i = Img.rows / 2; i < Img.rows; i++)
	{
		if (Img.at<uchar>(i, 0) == WHITE)FLoodFill(Img, i, 0);
		if (Img.at<uchar>(i, Img.cols - 1) == WHITE)FLoodFill(Img, i, Img.cols - 1);
	}
	for (int i = 0; i < Img.cols; i++)
	{
		if (Img.at<uchar>(0, i) == WHITE)FLoodFill(Img, 0, i);
		if (Img.at<uchar>(Img.rows - 1, i) == WHITE)FLoodFill(Img, Img.rows - 1, i);
	}

}
void revolve(Mat& Img, Mat& revolveMatrix, Size& NewSize)//旋转图片到合适的位置
{
	Mat Img_x;
	Sobel(Img, Img_x, -1, 0, 1, 5);
	vector<Vec2f>Lines;
	HoughLines(Img_x, Lines, 1, CV_PI / 180, 180);
	double angle = 0;
	int cnt = 0;
	for (auto i : Lines)
		if (i[1] > CV_PI * 17.0 / 36 && i[1] < CV_PI * 5.0 / 9 && i[0] < Img.rows / 3)
			//过滤直线：倾角在（85°,100°）
			angle += i[1], cnt++;
	angle /= cnt;

	revolveMatrix = getRotationMatrix2D(Point(Img.rows / 2, Img.cols / 2), 180 * angle / CV_PI - 90, 1);
	//逆时针旋转，获得旋转矩阵
	//根据旋转矩阵，计算旋转后图像的大小，为了防止边界的ISBN的数字丢失
	double cos = abs(revolveMatrix.at<double>(0, 0));
	double sin = abs(revolveMatrix.at<double>(0, 1));
	int nw = cos * Img.cols + sin * Img.rows;
	int nh = sin * Img.cols + cos * Img.rows;
	revolveMatrix.at<double>(0, 2) += (nw / 2 - Img.cols / 2);
	revolveMatrix.at<double>(1, 2) += (nh / 2 - Img.rows / 2);
	NewSize = { nw,nh };
}

int rowSum(Mat& Img, int rowi)//一行像素灰度值之和
{
	int sum = 0;
	for (int i = 0; i < Img.cols; i++)
		if (Img.at<uchar>(rowi, i) == 255)sum++;
	return sum;
}
int colSum(Mat& Img, int coli)//一列像素灰度值之和
{
	int sum = 0;
	for (int i = 0; i < Img.rows; i++)
		if (Img.at<uchar>(i, coli) == 255)sum++;
	return sum;
}
int allSum(Mat& Img)
{
	int sum = 0;
	for (int i = 0; i < Img.rows; i++)
		for (int j = 0; j < Img.cols; j++)
			if (Img.at<uchar>(i, j) == 255)sum++;
	return sum;
}

void getRow(Mat& Img)//获得ISBN的所在行
{
	double width = 1500;
	double height = width * Img.rows / Img.cols;
	resize(Img, Img, Size(width, height));
	toGrey(Img);
	denoising(Img, Img);
	toBinaryGraph(Img);
	Size newsize;
	Mat m;
	revolve(Img, m, newsize);
	warpAffine(Img, Img, m, newsize, INTER_LINEAR, 0);
	//找ISBN所在行的上下界
	int rowi = 0, Uper = 0, Lower = 0, Height = 0;
	while (rowi < Img.rows / 2)
	{
		while (rowi < Img.rows / 2 && rowSum(Img, rowi) < 10)rowi++;
		int upper = rowi;
		while (rowi < Img.rows / 2 && rowSum(Img, rowi) >= 10)rowi++;
		int lower = rowi;
		if (rowi < Img.rows / 2 && lower - upper > Height)
			Uper = upper, Lower = lower, Height = lower - upper;
	}
	if (Uper < Lower)
		Img = Mat(Img, Range(Uper, Lower), Range(0, Img.cols));
}


void getMinRec(Mat& Img)//获得最小矩阵
{
	int uper, lower;
	int rowi = 0;
	while (!rowSum(Img, rowi))rowi++;
	uper = rowi;
	rowi = Img.rows - 1;
	while (!rowSum(Img, rowi))rowi--;
	lower = rowi;
	if (uper < lower)
		Img = Mat(Img, Range(uper, lower), Range(0, Img.cols));

	int lefter, righter;
	int coli = 0;
	while (!colSum(Img, coli))coli++;
	lefter = coli;
	coli = Img.cols - 1;
	while (!colSum(Img, coli))coli--;
	righter = coli;
	if (lefter < righter)
		Img = Mat(Img, Range(0, Img.rows), Range(lefter, righter));

}

int main()
{
	int rtNums = 0, accNums = 0, sunNums = 0;//分别代表：正确的数量，被准确识别的字符的数量，要识别的字符的总和

	//1.读取 ISBN 图片

	vector<String> testImgFN;
	glob(testImgPath, testImgFN, false);//将所有 符合条件的图片 放入 testImgFN中
	int testImgNums = testImgFN.size();//得到图片总数

	clock_t start = clock();
	for (int i = 0; i < testImgNums; i++)
	{
		Mat Img = imread(testImgFN[i]);
		getRow(Img);
		getMinRec(Img);
		imshow("二值图像", Img);
		waitKey(6000);
	}
	clock_t end = clock();
	double timeSpend = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "共生成了：" << testImgNums << "张图片。 " << timeSpend / testImgNums << " Sec/prePic" << endl;

}
>>>>>>> c1abf96 (鍩虹鐩綍鍒涘缓)
