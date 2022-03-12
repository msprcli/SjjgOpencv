<<<<<<< HEAD
//#include<iostream>
//#include<string.h>
//#include<math.h>
//#include<vector>
//#include<opencv2/opencv.hpp>
//#include<opencv2\imgproc\imgproc_c.h>
//using namespace std;
//using namespace cv;
//
//void ToGrey(Mat& OrigImg)//转为灰度图
//{
//	int row = OrigImg.rows;
//	int col = OrigImg.cols;
//	Mat GreyImg;
//	GreyImg.create(row, col, CV_8UC1);
//	for (int i = 0; i < row; i++)
//		for (int j = 0; j < col; j++)
//		{
//			int B = OrigImg.at<Vec3b>(i, j)[0];
//			int G = OrigImg.at<Vec3b>(i, j)[1];
//			int R = OrigImg.at<Vec3b>(i, j)[2];
//			GreyImg.at<uchar>(i, j) = static_cast<uchar>(0.114 * B + 0.587 * G + 0.299 * R);
//		}
//	OrigImg = GreyImg;
//}
//
//int  MedianFilter(int* pixels, int size)//取中值
//{
//	sort(pixels, pixels + size);
//	return pixels[size / 2];
//}
//
//
//void RemoveNoise(Mat& OrigImg)//去除噪点，均值滤波法
//{
//	int row = OrigImg.rows;
//	int col = OrigImg.cols;
//	Mat ResImg = OrigImg.clone();
//	int pixels[9];
//	int move[9][2] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,0},{0,1},{1,-1},{1,0},{1,1} };
//	for (int i = 1; i < row - 1; i++)
//		for (int j = 1; j < col - 1; j++)
//		{
//			for (int k = 0; k < 9; k++)
//				pixels[k] = OrigImg.at<uchar>(i + move[k][0], j + move[k][1]);
//			ResImg.at<uchar>(i, j) = MedianFilter(pixels, 9);
//		}
//	OrigImg = ResImg;
//}
//
//
//int GetOtusThreshold(Mat& Img)//大津阈值法
//{
//	//绘制灰度直方图
//	int histogram[256] = { 0 };
//	for (int i = 0; i < Img.rows; i++)
//		for (int j = 0; j < Img.cols; j++)
//			histogram[Img.at<uchar>(i, j)]++;
//	//计算阈值
//	int T;//阈值
//	double Var = 0;//类间方差
//	int Total = Img.rows * Img.rows;
//	for (int i = 0; i < 255; i++)//枚举阈值
//	{
//		double FrontAvg = 0, FrontRate = 0, BackAvg = 0, BackRate = 0;
//		//前景色平均灰度，前景色像素占比，背景色平均灰度，背景色像素占比
//		for (int j = 0; j < 256; j++)
//			if (j <= i)FrontAvg += histogram[j] * j, FrontRate += histogram[j];//统计前景色的总灰度值和总像素数
//			else  BackAvg += histogram[j] * j, BackRate += histogram[j];//统计背景色的总灰度值和总像素数
//		FrontAvg /= FrontRate, FrontRate /= Total;//计算前景色的平均灰度值和像素占比
//		BackAvg /= BackRate, BackRate /= Total;//计算背景色的平均灰度值和像素占比
//		double Var1 = FrontRate * BackRate * (FrontAvg - BackAvg) * (FrontAvg - BackAvg);
//		if (Var1 > Var)Var = Var1, T = i;
//	}
//	return T;
//}
//
//
//void FLoodFill(Mat& Img, int x0, int y0)//八邻域填二值图充边界黑色部分为白色
//{
//	queue<pair<int, int>> q;
//	int move[8][2] = { {-1,-1},{-1,0},{-1,1},{0,1},{0,-1},{1,-1},{1,0},{1,1} };
//	Img.at<uchar>(x0, y0) = 0;
//	q.push({ x0,y0 });
//	while (q.size())
//	{
//		auto t = q.front();
//		q.pop();
//		for (int i = 0; i < 8; i++)
//		{
//			int x = t.first + move[i][0];
//			int y = t.second + move[i][1];
//			if (x >= 0 && x < Img.rows && y >= 0 &&
//				y < Img.cols && Img.at<uchar>(x, y) == 255)
//				q.push({ x,y }), Img.at<uchar>(x, y) = 0;
//		}
//	}
//}
//void ToBinary(Mat& Img, bool Invert = 1, bool FillEdege = 1)//将灰度图二值化，参数列表：二值化的图像，是否反相，是否填充边缘
//{
//	int Threshold = GetOtusThreshold(Img);
//	int front = 255, back = 0;//默认前景色为白色，底色为黑色
//	if (!Invert)front = 0, back = 255;
//	for (int i = 0; i < Img.rows; i++)
//		for (int j = 0; j < Img.cols; j++)
//			if (Img.at<uchar>(i, j) > Threshold)Img.at<uchar>(i, j) = back;
//			else Img.at<uchar>(i, j) = front;
//	if (!FillEdege)return;//不需要填充边缘，则返回
//	//防止IBSN受损，对边缘进行填充
//	for (int i = Img.rows / 2; i < Img.rows; i++)
//	{
//		if (Img.at<uchar>(i, 0) == 255)FLoodFill(Img, i, 0);
//		if (Img.at<uchar>(i, Img.cols - 1) == 255)FLoodFill(Img, i, Img.cols - 1);
//	}
//	for (int i = 0; i < Img.cols; i++)
//	{
//		if (Img.at<uchar>(0, i) == 255)FLoodFill(Img, 0, i);
//		if (Img.at<uchar>(Img.rows - 1, i) == 255)FLoodFill(Img, Img.rows - 1, i);
//	}
//
//}
//
//void Rotate(Mat& Img, Mat& RotateMatrix, Size& NewSize)//旋转图片到合适的位置
//{
//	Mat Img_x;
//	Sobel(Img, Img_x, -1, 0, 1, 5);
//	vector<Vec2f>Lines;
//	HoughLines(Img_x, Lines, 1, CV_PI / 180, 180);
//	double angle = 0;
//	int cnt = 0;
//	for (auto i : Lines)
//		if (i[1] > CV_PI * 17.0 / 36 && i[1] < CV_PI * 5.0 / 9 && i[0] < Img.rows / 3)
//			//过滤直线：倾角在（85°,100°）
//			angle += i[1], cnt++;
//	angle /= cnt;
//
//	RotateMatrix = getRotationMatrix2D(Point(Img.rows / 2, Img.cols / 2), 180 * angle / CV_PI - 90, 1);
//	//逆时针旋转，获得旋转矩阵，根据旋转矩阵，计算旋转后图像的大小，为了防止边界的ISBN的数字丢失
//	double cos = abs(RotateMatrix.at<double>(0, 0));
//	double sin = abs(RotateMatrix.at<double>(0, 1));
//	int nw = cos * Img.cols + sin * Img.rows;
//	int nh = sin * Img.cols + cos * Img.rows;
//	RotateMatrix.at<double>(0, 2) += (nw / 2 - Img.cols / 2);
//	RotateMatrix.at<double>(1, 2) += (nh / 2 - Img.rows / 2);
//	NewSize = { nw,nh };
//}
//
//int RowSum(Mat& Img, int RowIndex)//一行像素灰度值之和
//{
//	int sum = 0;
//	for (int i = 0; i < Img.cols; i++)
//		if (Img.at<uchar>(RowIndex, i) == 255)sum++;
//	return sum;
//}
//int ColSum(Mat& Img, int ColIndex)//一列像素灰度值之和
//{
//	int sum = 0;
//	for (int i = 0; i < Img.rows; i++)
//		if (Img.at<uchar>(i, ColIndex) == 255)sum++;
//	return sum;
//}
//int AllSum(Mat& Img)
//{
//	int sum = 0;
//	for (int i = 0; i < Img.rows; i++)
//		for (int j = 0; j < Img.cols; j++)
//			if (Img.at<uchar>(i, j) == 255)sum++;
//	return sum;
//}
//
//void GetISBNRow(Mat& Img)//获得ISBN的所在行
//{
//	double width = 1500;
//	double height = width * Img.rows / Img.cols;
//	resize(Img, Img, Size(width, height));
//	ToGrey(Img);
//	RemoveNoise(Img);
//	ToBinary(Img);
//	Size newsize;
//	Mat m;
//	Rotate(Img, m, newsize);
//	warpAffine(Img, Img, m, newsize, INTER_LINEAR, 0);
//	//找ISBN所在行的上下界
//	int RowIndex = 0, Upper_bound = 0, Lower_bound = 0, Height = 0;
//	while (RowIndex < Img.rows / 2)
//	{
//		while (RowIndex < Img.rows / 2 && RowSum(Img, RowIndex) < 10)RowIndex++;
//		int upper = RowIndex;
//		while (RowIndex < Img.rows / 2 && RowSum(Img, RowIndex) >= 10)RowIndex++;
//		int lower = RowIndex;
//		if (RowIndex < Img.rows / 2 && lower - upper > Height)
//			Upper_bound = upper, Lower_bound = lower, Height = lower - upper;
//	}
//	if (Upper_bound < Lower_bound)
//		Img = Mat(Img, Range(Upper_bound, Lower_bound), Range(0, Img.cols));
//}
//
//
//void GetMinRectangle(Mat& Img)//获得最小矩阵
//{
//	int Upper_bound, Lower_bound;
//	int RowIndex = 0;
//	while (!RowSum(Img, RowIndex))RowIndex++;
//	Upper_bound = RowIndex;
//	RowIndex = Img.rows - 1;
//	while (!RowSum(Img, RowIndex))RowIndex--;
//	Lower_bound = RowIndex;
//	if (Upper_bound < Lower_bound)
//		Img = Mat(Img, Range(Upper_bound, Lower_bound), Range(0, Img.cols));
//
//	int Left_bound, Right_bound;
//	int ColIndex = 0;
//	while (!ColSum(Img, ColIndex))ColIndex++;
//	Left_bound = ColIndex;
//	ColIndex = Img.cols - 1;
//	while (!ColSum(Img, ColIndex))ColIndex--;
//	Right_bound = ColIndex;
//	if (Left_bound < Right_bound)
//		Img = Mat(Img, Range(0, Img.rows), Range(Left_bound, Right_bound));
//
//}
//
//
//void SplitISBNRow(Mat& Img, vector<Mat>& Ch)//分割ISBN行，存到Ch中
//{
//	imshow("切分前的图片", Img);
//	waitKey(100);
//	int ColIndex = 0, Left_bound, Right_bound;
//	vector<Mat> ChCache;
//	int Height[30];
//	int Cnt = 0;
//	while (ColIndex < Img.cols)
//	{
//		while (ColIndex < Img.cols && !ColSum(Img, ColIndex))ColIndex++;
//		Left_bound = ColIndex;
//		while (ColIndex < Img.cols && ColSum(Img, ColIndex))ColIndex++;
//		Right_bound = ColIndex - 1;
//		if (Left_bound < Right_bound)
//		{
//			Mat Character = Mat(Img, Range(0, Img.rows), Range(Left_bound, Right_bound));
//			GetMinRectangle(Character);
//			ChCache.push_back(Character);
//			Height[Cnt++] = Character.rows;
//		}
//	}
//	int Mid = MedianFilter(Height, Cnt);
//	for (auto i : ChCache)
//		if (i.rows > 0.7 * Mid && i.rows < 1.3 * Mid) {
//			Ch.push_back(i);
//		}
//
//}
//
//bool Comp(pair<int, int>a, pair<int, int>b) {
//	return a.second < b.second;
//}
//int CalcImg(Mat inputImg) {
//	int nums = 0;
//	for (int i = 0; i < inputImg.rows; i++) {
//		for (int j = 0; j < inputImg.cols; j++) {
//			if (inputImg.at<uchar>(i, j) != 0) {
//				nums += inputImg.at<uchar>(i, j);
//			}
//		}
//	}
//	return nums;
//}
//
//char Compare(Mat TestImg, vector<pair<char, Mat>>& Mould) {
//	char best = '?';
//	double Max = 0;
//	resize(TestImg, TestImg, Size(40, 60));
//	//读取模板图片
//	string sampleImgPath = "./ISBN/数字样例/*";
//	vector<String> sampleImgFN;
//	glob(sampleImgPath, sampleImgFN, false);
//	int sampleImgNums = sampleImgFN.size();
//	pair<int, int>* nums = new pair<int, int>[sampleImgNums];//first 记录模板的索引号，second 记录两图像之差
//	for (int i = 0; i < sampleImgNums; i++) {
//		nums[i].first = i;
//		Mat numImg = imread(sampleImgFN[i], 0);
//		Mat delImg;
//		resize(numImg, numImg, Size(40, 60));
//		absdiff(numImg, TestImg, delImg);
//		nums[i].second = CalcImg(delImg);
//	}
//	sort(nums, nums + sampleImgNums, Comp);//选择差值最小的模板
//	//int index = nums[0].first / 2;
//	int index = nums[0].first;
//	return Mould[index].first;
//}
//
//int Judge(string& a, string& b)
//{
//	int f[50][50] = { 0 };
//	if (a[0] == b[0])f[0][0] = 1;
//	for (int i = 1; i < a.length(); i++)
//		for (int j = 1; j < b.length(); j++)
//			if (a[i] == b[j])f[i][j] = f[i - 1][j - 1] + 1;
//			else
//			{
//				f[i][j] = f[i - 1][j - 1];
//				f[i][j] = max(f[i][j], f[i - 1][j]);
//				f[i][j] = max(f[i][j], f[i][j - 1]);
//			}
//	return f[a.length() - 1][b.length() - 1];
//}
//
//void ReadTest(vector<String>& ImgFile)
//{
//	string testPath = "./ISBN/Train/*";
//	glob(testPath, ImgFile, false);
//}
//
//void ReadMould(vector<pair<char, Mat>>& Mould)
//{
//	vector<String> MouldPath;
//	string testPath = "./ISBN/数字样例/*";
//	glob(testPath, MouldPath, false);
//	for (auto i : MouldPath)
//	{
//		Mat MouldImg = imread(i, 0);
//		int p = i.find('\\');
//		ToBinary(MouldImg, 0, 0);
//		Mould.push_back({ i[p + 1],MouldImg });
//	}
//}
//
//int main()
//{
//	vector<String> TestFile;
//	ReadTest(TestFile);
//	vector<pair<char, Mat>> Mould;
//	ReadMould(Mould);
//	int OriginNumCnt = 0;
//	int RecogNumCnt = 0;
//	int Success = 0;
//	for (int i = 0; i < TestFile.size(); i++)
//	{
//		//从图中提取ISBN字符到Ch
//		Mat Img = imread(TestFile[i]);
//		imshow("当前图片", Img);
//		waitKey(100);
//		GetISBNRow(Img);
//		vector<Mat> Ch;
//		SplitISBNRow(Img, Ch);
//		//统计原始ISBN中数字的个数
//		int p1 = TestFile[i].find(' ');
//		int p2 = TestFile[i].rfind('.');
//		string OriginISBN = TestFile[i].substr(p1 + 1, p2 - p1 - 1);
//		for (int j = 0; j < OriginISBN.length(); j++)
//			if (isdigit(OriginISBN[j]))OriginNumCnt++;
//			else OriginISBN.erase(j, 1), j--;
//		//识别ISBN字符，统计成功识别出的ISBN中的数字个数
//		string RecogISBN;
//		int Start = -1, Pos = 0;
//		int num = 0;
//		for (auto j : Ch)
//		{
//			num++;
//			char c = Compare(j, Mould);
//			if (!isdigit(c))continue;
//			if (num > 4) {
//				if (Start == -1 && (c == '7' || c == '9'))Start = Pos;
//				RecogISBN += c;
//				Pos++;
//			}
//		}
//		if (Start != -1)
//		{
//			int len = 13;
//			if (RecogISBN[0] == '7')len = MIN(10, RecogISBN.length() - Start);
//			if (RecogISBN[0] == '9')len = MIN(13, RecogISBN.length() - Start);
//			RecogISBN = RecogISBN.substr(Start, len);
//		}
//		RecogNumCnt += Judge(OriginISBN, RecogISBN);
//		//判断该ISBN行是否识别成功
//		if (RecogISBN == OriginISBN) {
//			Success++, cout << OriginISBN << ' ' << RecogISBN << ' ' << "正确" << endl;
//		}
//		else {
//			cout << OriginISBN << ' ' << RecogISBN << ' ' << "错误" << endl;
//		}
//	}
//	cout << "共识别了：" << TestFile.size() << "张图片。" << endl;
//	cout << "正确率为:" << fixed << setprecision(2) << Success * 100.0 / TestFile.size() << "%  准确率为:" << fixed << setprecision(2) << RecogNumCnt * 100.0 / OriginNumCnt << "%" << endl;
//}
=======
//#include<iostream>
//#include<string.h>
//#include<math.h>
//#include<vector>
//#include<opencv2/opencv.hpp>
//#include<opencv2\imgproc\imgproc_c.h>
//using namespace std;
//using namespace cv;
//
//void ToGrey(Mat& OrigImg)//转为灰度图
//{
//	int row = OrigImg.rows;
//	int col = OrigImg.cols;
//	Mat GreyImg;
//	GreyImg.create(row, col, CV_8UC1);
//	for (int i = 0; i < row; i++)
//		for (int j = 0; j < col; j++)
//		{
//			int B = OrigImg.at<Vec3b>(i, j)[0];
//			int G = OrigImg.at<Vec3b>(i, j)[1];
//			int R = OrigImg.at<Vec3b>(i, j)[2];
//			GreyImg.at<uchar>(i, j) = static_cast<uchar>(0.114 * B + 0.587 * G + 0.299 * R);
//		}
//	OrigImg = GreyImg;
//}
//
//int  MedianFilter(int* pixels, int size)//取中值
//{
//	sort(pixels, pixels + size);
//	return pixels[size / 2];
//}
//
//
//void RemoveNoise(Mat& OrigImg)//去除噪点，均值滤波法
//{
//	int row = OrigImg.rows;
//	int col = OrigImg.cols;
//	Mat ResImg = OrigImg.clone();
//	int pixels[9];
//	int move[9][2] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,0},{0,1},{1,-1},{1,0},{1,1} };
//	for (int i = 1; i < row - 1; i++)
//		for (int j = 1; j < col - 1; j++)
//		{
//			for (int k = 0; k < 9; k++)
//				pixels[k] = OrigImg.at<uchar>(i + move[k][0], j + move[k][1]);
//			ResImg.at<uchar>(i, j) = MedianFilter(pixels, 9);
//		}
//	OrigImg = ResImg;
//}
//
//
//int GetOtusThreshold(Mat& Img)//大津阈值法
//{
//	//绘制灰度直方图
//	int histogram[256] = { 0 };
//	for (int i = 0; i < Img.rows; i++)
//		for (int j = 0; j < Img.cols; j++)
//			histogram[Img.at<uchar>(i, j)]++;
//	//计算阈值
//	int T;//阈值
//	double Var = 0;//类间方差
//	int Total = Img.rows * Img.rows;
//	for (int i = 0; i < 255; i++)//枚举阈值
//	{
//		double FrontAvg = 0, FrontRate = 0, BackAvg = 0, BackRate = 0;
//		//前景色平均灰度，前景色像素占比，背景色平均灰度，背景色像素占比
//		for (int j = 0; j < 256; j++)
//			if (j <= i)FrontAvg += histogram[j] * j, FrontRate += histogram[j];//统计前景色的总灰度值和总像素数
//			else  BackAvg += histogram[j] * j, BackRate += histogram[j];//统计背景色的总灰度值和总像素数
//		FrontAvg /= FrontRate, FrontRate /= Total;//计算前景色的平均灰度值和像素占比
//		BackAvg /= BackRate, BackRate /= Total;//计算背景色的平均灰度值和像素占比
//		double Var1 = FrontRate * BackRate * (FrontAvg - BackAvg) * (FrontAvg - BackAvg);
//		if (Var1 > Var)Var = Var1, T = i;
//	}
//	return T;
//}
//
//
//void FLoodFill(Mat& Img, int x0, int y0)//八邻域填二值图充边界黑色部分为白色
//{
//	queue<pair<int, int>> q;
//	int move[8][2] = { {-1,-1},{-1,0},{-1,1},{0,1},{0,-1},{1,-1},{1,0},{1,1} };
//	Img.at<uchar>(x0, y0) = 0;
//	q.push({ x0,y0 });
//	while (q.size())
//	{
//		auto t = q.front();
//		q.pop();
//		for (int i = 0; i < 8; i++)
//		{
//			int x = t.first + move[i][0];
//			int y = t.second + move[i][1];
//			if (x >= 0 && x < Img.rows && y >= 0 &&
//				y < Img.cols && Img.at<uchar>(x, y) == 255)
//				q.push({ x,y }), Img.at<uchar>(x, y) = 0;
//		}
//	}
//}
//void ToBinary(Mat& Img, bool Invert = 1, bool FillEdege = 1)//将灰度图二值化，参数列表：二值化的图像，是否反相，是否填充边缘
//{
//	int Threshold = GetOtusThreshold(Img);
//	int front = 255, back = 0;//默认前景色为白色，底色为黑色
//	if (!Invert)front = 0, back = 255;
//	for (int i = 0; i < Img.rows; i++)
//		for (int j = 0; j < Img.cols; j++)
//			if (Img.at<uchar>(i, j) > Threshold)Img.at<uchar>(i, j) = back;
//			else Img.at<uchar>(i, j) = front;
//	if (!FillEdege)return;//不需要填充边缘，则返回
//	//防止IBSN受损，对边缘进行填充
//	for (int i = Img.rows / 2; i < Img.rows; i++)
//	{
//		if (Img.at<uchar>(i, 0) == 255)FLoodFill(Img, i, 0);
//		if (Img.at<uchar>(i, Img.cols - 1) == 255)FLoodFill(Img, i, Img.cols - 1);
//	}
//	for (int i = 0; i < Img.cols; i++)
//	{
//		if (Img.at<uchar>(0, i) == 255)FLoodFill(Img, 0, i);
//		if (Img.at<uchar>(Img.rows - 1, i) == 255)FLoodFill(Img, Img.rows - 1, i);
//	}
//
//}
//
//void Rotate(Mat& Img, Mat& RotateMatrix, Size& NewSize)//旋转图片到合适的位置
//{
//	Mat Img_x;
//	Sobel(Img, Img_x, -1, 0, 1, 5);
//	vector<Vec2f>Lines;
//	HoughLines(Img_x, Lines, 1, CV_PI / 180, 180);
//	double angle = 0;
//	int cnt = 0;
//	for (auto i : Lines)
//		if (i[1] > CV_PI * 17.0 / 36 && i[1] < CV_PI * 5.0 / 9 && i[0] < Img.rows / 3)
//			//过滤直线：倾角在（85°,100°）
//			angle += i[1], cnt++;
//	angle /= cnt;
//
//	RotateMatrix = getRotationMatrix2D(Point(Img.rows / 2, Img.cols / 2), 180 * angle / CV_PI - 90, 1);
//	//逆时针旋转，获得旋转矩阵，根据旋转矩阵，计算旋转后图像的大小，为了防止边界的ISBN的数字丢失
//	double cos = abs(RotateMatrix.at<double>(0, 0));
//	double sin = abs(RotateMatrix.at<double>(0, 1));
//	int nw = cos * Img.cols + sin * Img.rows;
//	int nh = sin * Img.cols + cos * Img.rows;
//	RotateMatrix.at<double>(0, 2) += (nw / 2 - Img.cols / 2);
//	RotateMatrix.at<double>(1, 2) += (nh / 2 - Img.rows / 2);
//	NewSize = { nw,nh };
//}
//
//int RowSum(Mat& Img, int RowIndex)//一行像素灰度值之和
//{
//	int sum = 0;
//	for (int i = 0; i < Img.cols; i++)
//		if (Img.at<uchar>(RowIndex, i) == 255)sum++;
//	return sum;
//}
//int ColSum(Mat& Img, int ColIndex)//一列像素灰度值之和
//{
//	int sum = 0;
//	for (int i = 0; i < Img.rows; i++)
//		if (Img.at<uchar>(i, ColIndex) == 255)sum++;
//	return sum;
//}
//int AllSum(Mat& Img)
//{
//	int sum = 0;
//	for (int i = 0; i < Img.rows; i++)
//		for (int j = 0; j < Img.cols; j++)
//			if (Img.at<uchar>(i, j) == 255)sum++;
//	return sum;
//}
//
//void GetISBNRow(Mat& Img)//获得ISBN的所在行
//{
//	double width = 1500;
//	double height = width * Img.rows / Img.cols;
//	resize(Img, Img, Size(width, height));
//	ToGrey(Img);
//	RemoveNoise(Img);
//	ToBinary(Img);
//	Size newsize;
//	Mat m;
//	Rotate(Img, m, newsize);
//	warpAffine(Img, Img, m, newsize, INTER_LINEAR, 0);
//	//找ISBN所在行的上下界
//	int RowIndex = 0, Upper_bound = 0, Lower_bound = 0, Height = 0;
//	while (RowIndex < Img.rows / 2)
//	{
//		while (RowIndex < Img.rows / 2 && RowSum(Img, RowIndex) < 10)RowIndex++;
//		int upper = RowIndex;
//		while (RowIndex < Img.rows / 2 && RowSum(Img, RowIndex) >= 10)RowIndex++;
//		int lower = RowIndex;
//		if (RowIndex < Img.rows / 2 && lower - upper > Height)
//			Upper_bound = upper, Lower_bound = lower, Height = lower - upper;
//	}
//	if (Upper_bound < Lower_bound)
//		Img = Mat(Img, Range(Upper_bound, Lower_bound), Range(0, Img.cols));
//}
//
//
//void GetMinRectangle(Mat& Img)//获得最小矩阵
//{
//	int Upper_bound, Lower_bound;
//	int RowIndex = 0;
//	while (!RowSum(Img, RowIndex))RowIndex++;
//	Upper_bound = RowIndex;
//	RowIndex = Img.rows - 1;
//	while (!RowSum(Img, RowIndex))RowIndex--;
//	Lower_bound = RowIndex;
//	if (Upper_bound < Lower_bound)
//		Img = Mat(Img, Range(Upper_bound, Lower_bound), Range(0, Img.cols));
//
//	int Left_bound, Right_bound;
//	int ColIndex = 0;
//	while (!ColSum(Img, ColIndex))ColIndex++;
//	Left_bound = ColIndex;
//	ColIndex = Img.cols - 1;
//	while (!ColSum(Img, ColIndex))ColIndex--;
//	Right_bound = ColIndex;
//	if (Left_bound < Right_bound)
//		Img = Mat(Img, Range(0, Img.rows), Range(Left_bound, Right_bound));
//
//}
//
//
//void SplitISBNRow(Mat& Img, vector<Mat>& Ch)//分割ISBN行，存到Ch中
//{
//	imshow("切分前的图片", Img);
//	waitKey(100);
//	int ColIndex = 0, Left_bound, Right_bound;
//	vector<Mat> ChCache;
//	int Height[30];
//	int Cnt = 0;
//	while (ColIndex < Img.cols)
//	{
//		while (ColIndex < Img.cols && !ColSum(Img, ColIndex))ColIndex++;
//		Left_bound = ColIndex;
//		while (ColIndex < Img.cols && ColSum(Img, ColIndex))ColIndex++;
//		Right_bound = ColIndex - 1;
//		if (Left_bound < Right_bound)
//		{
//			Mat Character = Mat(Img, Range(0, Img.rows), Range(Left_bound, Right_bound));
//			GetMinRectangle(Character);
//			ChCache.push_back(Character);
//			Height[Cnt++] = Character.rows;
//		}
//	}
//	int Mid = MedianFilter(Height, Cnt);
//	for (auto i : ChCache)
//		if (i.rows > 0.7 * Mid && i.rows < 1.3 * Mid) {
//			Ch.push_back(i);
//		}
//
//}
//
//bool Comp(pair<int, int>a, pair<int, int>b) {
//	return a.second < b.second;
//}
//int CalcImg(Mat inputImg) {
//	int nums = 0;
//	for (int i = 0; i < inputImg.rows; i++) {
//		for (int j = 0; j < inputImg.cols; j++) {
//			if (inputImg.at<uchar>(i, j) != 0) {
//				nums += inputImg.at<uchar>(i, j);
//			}
//		}
//	}
//	return nums;
//}
//
//char Compare(Mat TestImg, vector<pair<char, Mat>>& Mould) {
//	char best = '?';
//	double Max = 0;
//	resize(TestImg, TestImg, Size(40, 60));
//	//读取模板图片
//	string sampleImgPath = "./ISBN/数字样例/*";
//	vector<String> sampleImgFN;
//	glob(sampleImgPath, sampleImgFN, false);
//	int sampleImgNums = sampleImgFN.size();
//	pair<int, int>* nums = new pair<int, int>[sampleImgNums];//first 记录模板的索引号，second 记录两图像之差
//	for (int i = 0; i < sampleImgNums; i++) {
//		nums[i].first = i;
//		Mat numImg = imread(sampleImgFN[i], 0);
//		Mat delImg;
//		resize(numImg, numImg, Size(40, 60));
//		absdiff(numImg, TestImg, delImg);
//		nums[i].second = CalcImg(delImg);
//	}
//	sort(nums, nums + sampleImgNums, Comp);//选择差值最小的模板
//	//int index = nums[0].first / 2;
//	int index = nums[0].first;
//	return Mould[index].first;
//}
//
//int Judge(string& a, string& b)
//{
//	int f[50][50] = { 0 };
//	if (a[0] == b[0])f[0][0] = 1;
//	for (int i = 1; i < a.length(); i++)
//		for (int j = 1; j < b.length(); j++)
//			if (a[i] == b[j])f[i][j] = f[i - 1][j - 1] + 1;
//			else
//			{
//				f[i][j] = f[i - 1][j - 1];
//				f[i][j] = max(f[i][j], f[i - 1][j]);
//				f[i][j] = max(f[i][j], f[i][j - 1]);
//			}
//	return f[a.length() - 1][b.length() - 1];
//}
//
//void ReadTest(vector<String>& ImgFile)
//{
//	string testPath = "./ISBN/Train/*";
//	glob(testPath, ImgFile, false);
//}
//
//void ReadMould(vector<pair<char, Mat>>& Mould)
//{
//	vector<String> MouldPath;
//	string testPath = "./ISBN/数字样例/*";
//	glob(testPath, MouldPath, false);
//	for (auto i : MouldPath)
//	{
//		Mat MouldImg = imread(i, 0);
//		int p = i.find('\\');
//		ToBinary(MouldImg, 0, 0);
//		Mould.push_back({ i[p + 1],MouldImg });
//	}
//}
//
//int main()
//{
//	vector<String> TestFile;
//	ReadTest(TestFile);
//	vector<pair<char, Mat>> Mould;
//	ReadMould(Mould);
//	int OriginNumCnt = 0;
//	int RecogNumCnt = 0;
//	int Success = 0;
//	for (int i = 0; i < TestFile.size(); i++)
//	{
//		//从图中提取ISBN字符到Ch
//		Mat Img = imread(TestFile[i]);
//		imshow("当前图片", Img);
//		waitKey(100);
//		GetISBNRow(Img);
//		vector<Mat> Ch;
//		SplitISBNRow(Img, Ch);
//		//统计原始ISBN中数字的个数
//		int p1 = TestFile[i].find(' ');
//		int p2 = TestFile[i].rfind('.');
//		string OriginISBN = TestFile[i].substr(p1 + 1, p2 - p1 - 1);
//		for (int j = 0; j < OriginISBN.length(); j++)
//			if (isdigit(OriginISBN[j]))OriginNumCnt++;
//			else OriginISBN.erase(j, 1), j--;
//		//识别ISBN字符，统计成功识别出的ISBN中的数字个数
//		string RecogISBN;
//		int Start = -1, Pos = 0;
//		int num = 0;
//		for (auto j : Ch)
//		{
//			num++;
//			char c = Compare(j, Mould);
//			if (!isdigit(c))continue;
//			if (num > 4) {
//				if (Start == -1 && (c == '7' || c == '9'))Start = Pos;
//				RecogISBN += c;
//				Pos++;
//			}
//		}
//		if (Start != -1)
//		{
//			int len = 13;
//			if (RecogISBN[0] == '7')len = MIN(10, RecogISBN.length() - Start);
//			if (RecogISBN[0] == '9')len = MIN(13, RecogISBN.length() - Start);
//			RecogISBN = RecogISBN.substr(Start, len);
//		}
//		RecogNumCnt += Judge(OriginISBN, RecogISBN);
//		//判断该ISBN行是否识别成功
//		if (RecogISBN == OriginISBN) {
//			Success++, cout << OriginISBN << ' ' << RecogISBN << ' ' << "正确" << endl;
//		}
//		else {
//			cout << OriginISBN << ' ' << RecogISBN << ' ' << "错误" << endl;
//		}
//	}
//	cout << "共识别了：" << TestFile.size() << "张图片。" << endl;
//	cout << "正确率为:" << fixed << setprecision(2) << Success * 100.0 / TestFile.size() << "%  准确率为:" << fixed << setprecision(2) << RecogNumCnt * 100.0 / OriginNumCnt << "%" << endl;
//}
>>>>>>> c1abf96 (鍩虹鐩綍鍒涘缓)
