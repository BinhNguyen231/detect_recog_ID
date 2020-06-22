#pragma once

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include<math.h>
#include <baseapi.h>
#include <allheaders.h>
#include <sstream>
#include <time.h>
#include <opencv2/dnn/dnn.hpp>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;
using namespace dlib;

template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;

// A 3x3 conv layer that doesn't do any downsampling
template <long num_filters, typename SUBNET> using con3 = con<num_filters, 3, 3, 1, 1, SUBNET>;

// Now we can define the 8x downsampling block in terms of conv5d blocks.  We
// also use relu and batch normalization in the standard way.
template <typename SUBNET> using downsampler_quoc_huy = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32, SUBNET>>>>>>>>>;

// The rest of the network will be 3x3 conv layers with batch normalization and
// relu.  So we define the 3x3 block we will use here.
template <typename SUBNET> using rcon3 = relu<bn_con<con3<32, SUBNET>>>;

// Finally, we define the entire network.   The special input_rgb_image_pyramid
// layer causes the network to operate over a spatial pyramid, making the detector
// scale invariant.  
using net_type_ = loss_mmod<con<1, 6, 6, 1, 1, rcon3<rcon3<rcon3<downsampler_quoc_huy<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

class ID_card
{
private:
	cv::Point face;
	cv::Point quoc_huy;
	double angle;
	cv::Rect so;
	//cv::Rect roi1;
	cv::Rect roi_to_detect_So;
	cv::Rect bb_id;
	cv::Rect bb_hoten;
	cv::Rect bb_dob;
	string hoten;
	string dob;
	string id;
	double ratio;
	double d_in_img_1440;
	Mat img, img_800, img_1440, img_to_detect_So;
	matrix<dlib::rgb_pixel> img_dlib, img_dlib_detect_So;
	string path_to_image;
public:
	ID_card(string path);
	bool detectFace(cv::dnn::Net net);
	bool detectQuochuy(net_type_ net);
	bool detectSo(net_type_ net);
	bool checkRotateImage(cv::dnn::Net , net_type_ );
	void rotate();
	std::vector<cv::Rect> detectFeature(cv::Mat gray);
	void recognizeFeature(cv::dnn::Net net_face, net_type_ net_so, net_type_ net_quohuy);
	void recogFeatureSolution1();
	void recogFeatureSolution2();
	//void get_angle_rotate(float x1, float y1, float x2, float y2);
	void recogText(cv::Mat gray, string mode);
	void show_img();
	~ID_card();
};


