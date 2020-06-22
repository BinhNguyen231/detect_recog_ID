#include "ID_card.h"

using namespace std;

ID_card::ID_card(string path)
{
	// Load and resize image
	path_to_image = path;
	img = imread(path, IMREAD_COLOR);
	int h = img.rows;
	int w = img.cols;
	ratio = 1.0 * h / w;
	resize(img, img_800, Size(800, int(800 * ratio)));
	resize(img, img_1440, Size(1440, int(1440 * ratio)));
	Mat img_rgb;
	cvtColor(img_800, img_rgb, COLOR_BGR2RGB);
	assign_image(img_dlib, cv_image<rgb_pixel>(img_rgb));
	//Khoi tao cac bien
	face = cv::Point(-1, -1);
	quoc_huy = cv::Point(-1, -1);
	angle = 0.0;
	so = cv::Rect(-1, -1, -1, -1);
	roi_to_detect_So = cv::Rect(-1, -1, -1, -1);
	bb_id = cv::Rect(-1, -1, -1, -1);
	bb_hoten = cv::Rect(-1, -1, -1, -1);
	bb_dob = cv::Rect(-1, -1, -1, -1);
	hoten = "";
	dob = "";
	id = "";
}

void ID_card::recognizeFeature(cv::dnn::Net net_face, net_type_ net_so, net_type_ net_quohuy)
{
	bool check = checkRotateImage(net_face, net_quohuy);
	if (!check)
	{
		cout << "Khong nhan dien duoc khuon mat va quoc huy" << endl;
		cout << "Chuyen qua anh tiep theo ..." << endl;
		return;
	}
	stringstream ss_1440;
	string sstr = path_to_image.substr(path_to_image.length() - 6, 6);
	ss_1440 << "result/img_1440_" << sstr;
	imwrite(ss_1440.str(), img_1440);
	bool check_so = detectSo(net_so);
	if (check_so)
	{
		recogFeatureSolution1();
	}
	else{
		recogFeatureSolution2();
	}
	

}
bool ID_card::checkRotateImage(cv::dnn::Net net_face, net_type_ net_quochuy)
{
	int count = 0;
	while (1)
	{
		if (count == 5)
			return false;

		bool check_face = detectFace(net_face);
		if (!check_face)
			return false;
		bool check_quochuy = detectQuochuy(net_quochuy);
		if (!check_quochuy)
			return false;
		count++;
		int x1 = quoc_huy.x;
		int y1 = quoc_huy.y;
		int x2 = face.x;
		int y2 = face.y;
		if (abs(x2 - x1) > int(img_800.cols / 200) || (y1 >= y2))
		{
			const double PI = 3.1415926536;
			float w = x1 - x2;
			float h = y1 - y2;
			float ang = atan((w*1.0) / h * 1.0);
			angle = (ang*180.0) / PI;
			if (y1 >= y2)
				angle += 180;
			cout << "angle " << count << " : " << angle << endl;
			rotate();
		}
		else {
			double d = sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
			double ty_le = 1440.0 / 800.0;
			d = d * ty_le; x1 *= ty_le; y1 *= ty_le;
			d_in_img_1440 = d;
			int x = int(x1 + d * 0.75);
			int y = y1;
			int w = min(int(3 * d), int(img_1440.cols - x));
			int h = min(int(1.25 * d), int(img.rows - y));
			roi_to_detect_So = Rect(x, y, w, h);
			Mat tmp_mat(img_1440, roi_to_detect_So);
			tmp_mat.copyTo(img_to_detect_So);
			cvtColor(img_to_detect_So, tmp_mat, COLOR_BGR2RGB);
			assign_image(img_dlib_detect_So, cv_image<rgb_pixel>(tmp_mat));
			break;
		}
	}
	cout << "So lan detect: " << count << endl;
	return true;
}

bool ID_card::detectFace(cv::dnn::Net net)
{
	Mat img_for_face;
	resize(img_800, img_for_face, Size(512, 512));
	net.setInput(cv::dnn::blobFromImage(img_for_face, 1.0f, cv::Size(512, 512),
		cv::Scalar(104.0, 177.0, 123.0), false));
	cv::Mat out_face = net.forward();
	bool tmp = 0;
	cv::Mat detected_face(out_face.size[2], out_face.size[3], CV_32F, out_face.ptr<float>());
	if (detected_face.rows >= 1)
	{
		float confidence = detected_face.at<float>(0, 2);
		if (confidence > 0.5) {
			std::cout << "Detected face in cmnd " << endl;
			int face_x1 = static_cast<int>(detected_face.at<float>(0, 3) * img_800.cols);
			int face_y1 = static_cast<int>(detected_face.at<float>(0, 4) * img_800.rows);
			int face_x2 = static_cast<int>(detected_face.at<float>(0, 5) * img_800.cols);
			int face_y2 = static_cast<int>(detected_face.at<float>(0, 6) * img_800.rows);
			float center_face_x = (face_x1 + face_x2) / 2.0;
			float center_face_y = (face_y1 + face_y2) / 2.0;
			face = Point(center_face_x, center_face_y);
			tmp = 1;
		}
	}
	return tmp;
}

bool ID_card::detectQuochuy(net_type_ net)
{
	auto dets_quoc_huy = net(img_dlib);
	bool tmp = 0;
	if (dets_quoc_huy.size() >= 1)
	{
		std::cout << "Detected quoc huy " << endl;
		float center_quoc_huy_x, center_quoc_huy_y = 0.0;
		center_quoc_huy_x = (dets_quoc_huy[0].rect.left() + dets_quoc_huy[0].rect.right()) / 2.0;
		center_quoc_huy_y = (dets_quoc_huy[0].rect.top() + dets_quoc_huy[0].rect.bottom()) / 2.0;
		/*int t, l, r, b;
		l = int(dets_quoc_huy[0].rect.left());
		t = int(dets_quoc_huy[0].rect.top());
		r = int(dets_quoc_huy[0].rect.right());
		b = int(dets_quoc_huy[0].rect.bottom());*/
		quoc_huy = cv::Point(center_quoc_huy_x, center_quoc_huy_y);
		/*cv::rectangle(img_800, Point(l, t), Point(r, b), cv::Scalar(0, 255, 0));
		imshow("img_800", img_800);
		waitKey(0);*/
		tmp = 1;
	}
	return tmp;
}

void ID_card::rotate()
{
	// rotate img_1440
	cv::Point2f pc(img_1440.cols / 2., img_1440.rows / 2.);
	cv::Mat r = cv::getRotationMatrix2D(pc, 0 - angle, 1.0);
	cv::warpAffine(img_1440, img_1440, r, img_1440.size());


	//rotate img_800
	//imshow("8001", img_800);
	cv::Point2f pc2(img_800.cols / 2., img_800.rows / 2.);
	cv::Mat r2 = cv::getRotationMatrix2D(pc2, 0 - angle, 1.0);
	cv::warpAffine(img_800, img_800, r2, img_800.size());
	//imshow("8002", img_800);
	//waitKey();
	cv::Mat tmp;
	cvtColor(img_800, tmp, COLOR_BGR2RGB);
	assign_image(img_dlib, cv_image<rgb_pixel>(tmp));
}

bool ID_card::detectSo(net_type_ net)
{
	auto dets_so = net(img_dlib_detect_So);
	bool tmp = 0;
	if (dets_so.size() >= 1) {
		cout << "Detected 'So'" << endl;
		int l, t, r, b, width, height = 0;
		l = dets_so[0].rect.left();
		t = dets_so[0].rect.top();
		r = dets_so[0].rect.right();
		b = dets_so[0].rect.bottom();
		cv::Rect bb_so;
		bb_so.x = l;
		bb_so.y = t;
		bb_so.width = r - l;
		bb_so.height = b - t;
		so = bb_so;
		tmp = 1;
	}
	return tmp;
}

void ID_card::recogFeatureSolution1()
{
	Rect roi_id, roi_hoten_dob;
	roi_id.x = so.x + int(so.width * 1.75);
	roi_id.y = so.y - int(so.height * 0.25);
	roi_id.width = min(int(1.2 * d_in_img_1440), img_to_detect_So.cols - roi_id.x);
	roi_id.height = min(int(so.height * 1.75), img_to_detect_So.rows - roi_id.y);

	//compute roi2 chua ho_ten va ngay_sinh
	roi_hoten_dob.x = so.x + so.width / 2;
	roi_hoten_dob.y = so.y + int(so.height*1.3);
	roi_hoten_dob.width = min(int(2 * d_in_img_1440), img_to_detect_So.cols - roi_hoten_dob.x);
	roi_hoten_dob.height = min(int(d_in_img_1440 * 0.8), img_to_detect_So.rows - roi_hoten_dob.y);

	cvtColor(img_to_detect_So, img_to_detect_So, cv::COLOR_BGR2GRAY);
	cv::Mat feature(img_to_detect_So, roi_hoten_dob);
	//imshow("feature", feature);
	//waitKey();
	int height = feature.rows;
	int width = feature.cols;
	resize(feature, feature, Size(int(150 * (width * 1.0 / height)), 150));
	std::vector<Rect> boundRects = this->detectFeature(feature);
	if (boundRects.size() != 2)
	{
		std::cout << "No detected 2 feature" << endl;
		cout << "Detected " << boundRects.size() << "features" << endl;
		return;
	}
	cv::Mat DoB(feature, boundRects[0]);
	cv::Mat Name(feature, boundRects[1]);
	cv::Mat ID(img_to_detect_So, roi_id);

	stringstream ss_id, ss_name, ss_dob;
	string sstr = path_to_image.substr(path_to_image.length() - 6, 6);
	ss_id << "crop_roi/Id_" << sstr;
	ss_name << "crop_roi/Name_" << sstr;
	ss_dob << "crop_roi/Dob_" << sstr;
	imwrite(ss_id.str(), ID);
	imwrite(ss_name.str(), Name);
	imwrite(ss_dob.str(), DoB);

	//tesseract tect recognition
	this->recogText(ID, "id");
	this->recogText(Name, "name");
	this->recogText(DoB, "dob");
}

void ID_card::recogFeatureSolution2()
{
	cout << "Khong nhan dien duoc 'So' " << endl;
	cout << "Su dung phuong phap 2..." << endl;
	Rect roi_id_hoten_dob;
	roi_id_hoten_dob.x = int(d_in_img_1440 * 0.35);
	roi_id_hoten_dob.y = 10;
	roi_id_hoten_dob.width = min(int(2.25 * d_in_img_1440), img_to_detect_So.cols - roi_id_hoten_dob.x);
	roi_id_hoten_dob.height = min(int(d_in_img_1440), img_to_detect_So.rows - roi_id_hoten_dob.y);

	cvtColor(img_to_detect_So, img_to_detect_So, cv::COLOR_BGR2GRAY);
	cv::Mat feature(img_to_detect_So, roi_id_hoten_dob);
	//imshow("feature", feature);
	//waitKey();
	int height = feature.rows;
	int width = feature.cols;
	resize(feature, feature, Size(int(200 * (width * 1.0 / height)), 200));
	std::vector<Rect> boundRects = this->detectFeature(feature);
	if (boundRects.size() != 3)
	{
		std::cout << "No detected 3 feature" << endl;
		cout << "Detected " << boundRects.size() << "features" << endl;
		return;
	}
	cv::Mat DoB(feature, boundRects[0]);
	cv::Mat Name(feature, boundRects[1]);
	cv::Mat ID(feature, boundRects[2]);

	stringstream ss_id, ss_name, ss_dob;
	string sstr = path_to_image.substr(path_to_image.length() - 6, 6);
	ss_id << "crop_roi/Id_" << sstr;
	ss_name << "crop_roi/Name_" << sstr;
	ss_dob << "crop_roi/Dob_" << sstr;
	imwrite(ss_id.str(), ID);
	imwrite(ss_name.str(), Name);
	imwrite(ss_dob.str(), DoB);

	//tesseract tect recognition
	this->recogText(ID, "id");
	this->recogText(Name, "name");
	this->recogText(DoB, "dob");
}

std::vector<cv::Rect> ID_card::detectFeature(cv::Mat gray)
{
	int w = gray.cols;
	int h = gray.rows;
	Mat rectKernel, sqKernel;
	rectKernel = getStructuringElement(cv::MORPH_RECT, Size(23, 5));
	sqKernel = getStructuringElement(cv::MORPH_RECT, Size(7, 7));

	Mat top_hat;
	Mat	thresh(gray.rows, gray.cols, CV_8UC1);
	Mat gradX(gray.rows, gray.cols, CV_32F);
	morphologyEx(gray, top_hat, MORPH_TOPHAT, rectKernel);
	Sobel(top_hat, gradX, CV_32F, 1, 0, -1);
	gradX = cv::abs(gradX);
	double minVal, maxVal;
	minMaxLoc(gradX, &minVal, &maxVal);
	for (int i = 0; i < gradX.rows; i++)
	{
		for (int j = 0; j < gradX.cols; j++)
		{
			int p = gradX.at<float>(i, j);
			thresh.at<uchar>(i, j) = int(255 * ((p - minVal) *1.0 / (maxVal - minVal)));
		}
	}

	cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, rectKernel);
	threshold(thresh, thresh, 0, 255, THRESH_OTSU);
	cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, sqKernel);
	stringstream ss_thresh;
	string sstr = path_to_image.substr(path_to_image.length() - 6, 6);
	ss_thresh << "result/Thresh_" << sstr;
	cv::imwrite(ss_thresh.str(), thresh);
	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

	findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	std::vector<Rect> boundRect_tmp(contours.size());
	std::vector<Rect> boundRects;
	int index = 0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		boundRect_tmp[i] = boundingRect(contours[i]);
		int x_, y_, w_, h_;
		x_ = boundRect_tmp[i].x;
		y_ = boundRect_tmp[i].y;
		w_ = boundRect_tmp[i].width;
		h_ = boundRect_tmp[i].height;
		double ar = w_ * 1.0 / h_;
		if (ar > 4.5 && ar < 25) {
			if ((w_ > w / 6.0) && (h_ > h / 10.0) && (h_ < h / 2.0))
			{
				cv::Rect roi;
				int margin = 3;
				roi.x = max(int(x_ - margin), 0);
				roi.y = max(y_ - margin * 2 , 0);
				roi.width = min(w_ + margin * 8, gray.cols - roi.x);
				roi.height = min(h_ + margin * 4, gray.rows - roi.y);
				//if (index == 2) // ID
				//{
				//	roi.x = roi.x - margin * 5;
				//	roi.width = roi.width + margin * 5;
				//}
				boundRects.push_back(roi);
				//stringstream ss;
				//string sstr = path.substr(path.length() - 6, 6);
				//ss << "result/Feature_" << sstr;
				//imwrite(ss.str(), gray);
				index++;
			}
		}
	}
	return boundRects;
}




void ID_card:: recogText(cv::Mat gray,  string mode)
{
	id = hoten = dob = "";
	Mat thresh;
	/*threshold(gray, thresh, 250, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
	GaussianBlur(thresh, thresh, cv::Size(3, 3), 0);
	threshold(thresh, thresh, 250, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);*/


	Mat kernel, gray_blur, gray_erode, gray_mor, gray_bitwise;
	kernel = getStructuringElement(cv::MORPH_RECT, Size(3, 3));

	bilateralFilter(gray, gray_blur, 3, 75, 75);
	erode(gray_blur, gray_erode, kernel);
	bitwise_and(gray_blur, gray_erode, gray_bitwise);
	//imshow("bitwise", gray_bitwise);
	threshold(gray_bitwise, gray_mor, 250, 255, THRESH_BINARY + THRESH_OTSU);
	//imshow("gray_mor", gray_mor);
	morphologyEx(gray_mor, thresh, MORPH_CLOSE, kernel, Point(-1, -1), 1);
	//imshow("last_gray_mor", gray_mor);
	dilate(thresh, thresh, kernel);

	//imshow("thresh", thresh);
	int pad_size[2];
	pad_size[0] = thresh.cols + 30 * 2;
	pad_size[1] = thresh.rows + 60 * 2;
	cv::Mat thresh_pad(1, pad_size, CV_32FC1, cv::Scalar(255));
	cv::copyMakeBorder(thresh, thresh_pad, 60, 60, 30, 30, cv::BORDER_CONSTANT, Scalar(255));
	//imshow("thresh_pad", thresh_pad);
	//waitKey();

	// Run Tesseract OCR on image
	tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
	if (mode == "id" || mode == "dob") {
		ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
	}
	else {
		ocr->Init(NULL, "vie", tesseract::OEM_LSTM_ONLY);
	}
	ocr->SetPageSegMode(tesseract::PSM_AUTO);
	ocr->SetImage(thresh_pad.data, thresh_pad.cols, thresh_pad.rows, 1, thresh_pad.step);
	string outText = ocr->GetUTF8Text();
	string outText_processed = "";
	stringstream ss;
	string sstr = path_to_image.substr(path_to_image.length() - 6, 6);
	if (mode == "id")
	{
		for (int j = 0; j < outText.length(); j++)
		{
			if (outText[j] <= '9' && outText[j] >= '0')
				outText_processed += outText[j];
		}
		ss << "result/Tessseract_id_" << sstr;
		id = outText_processed;
	}
	else if (mode == "name")
	{
		for (int j = 0; j < outText.length(); j++)
		{
			int tmp = int(outText[j]);
			if (tmp < 0 || tmp == ' ' || ((tmp >= 'A') && (tmp <= 'Z')))
			{
				outText_processed += outText[j];
			}
		}
		ss << "result/Tessseract_name_" << sstr;
		hoten = outText_processed;
	}
	else if (mode == "dob")
	{
		for (int j = 0; j < outText.length(); j++)
		{
			if ((outText[j] <= '9' && outText[j] >= '0') || (outText[j] == '-') || outText[j] == ' ')
				outText_processed += outText[j];
		}
		ss << "result/Tessseract_dob_" << sstr;
		dob = outText_processed;
	}
	ocr->End();

	imwrite(ss.str(), thresh_pad);

	ofstream resultFile;
	resultFile.open("result.txt", ios::out | ios::app);
	resultFile << ss.str() << " : " << outText_processed << "\n";
	resultFile.close();
}

void ID_card::show_img()
{
	imshow("img800", img_800);
	imshow("img1440", img_1440);
	imshow("img_so", img_to_detect_So);
	waitKey();
}

ID_card::~ID_card()
{
}
