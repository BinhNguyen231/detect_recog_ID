// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
	This example shows how to train a CNN based object detector using dlib's
	loss_mmod loss layer.  This loss layer implements the Max-Margin Object
	Detection loss as described in the paper:
		Max-Margin Object Detection by Davis E. King (http://arxiv.org/abs/1502.00046).
	This is the same loss used by the popular SVM+HOG object detector in dlib
	(see fhog_object_detector_ex.cpp) except here we replace the HOG features
	with a CNN and train the entire detector end-to-end.  This allows us to make
	much more powerful detectors.

	It would be a good idea to become familiar with dlib's DNN tooling before
	reading this example.  So you should read dnn_introduction_ex.cpp and
	dnn_introduction2_ex.cpp before reading this example program.

	Just like in the fhog_object_detector_ex.cpp example, we are going to train
	a simple face detector based on the very small training dataset in the
	examples/faces folder.  As we will see, even with this small dataset the
	MMOD method is able to make a working face detector.  However, for real
	applications you should train with more data for an even better result.
*/


#include "ID_card.h"



using namespace std;
using namespace dlib;
using namespace cv;

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

int main(int argc, char** argv) try
{

	net_type_ net_quochuy;
	net_type_ net_so;

	deserialize("dlib_model/mmod_network_quoc_huy.dat") >> net_quochuy;
	deserialize("dlib_model/mmod_network_so_thay_gui.dat") >> net_so;

	//Load model detect face from caffe 
	cv::dnn::Net net_face = cv::dnn::readNetFromCaffe("caffe_model/VGG16_SSD_512.prototxt", 
		"caffe_model/VGG16_SSD_512.caffemodel");

	const std::string folder_imgs = "test_images/*.jpg";

	std::vector<string> file_names;

	cv::glob(folder_imgs, file_names);
	cout << "Total file name is: " << file_names.size() << endl;

	for (auto f : file_names)
	{
		// clock start
		clock_t start = clock();

		cout << endl << "Processing " << f << " ..." << endl;
		ID_card *ID = new ID_card(f);

		//Detect and Recognition feature
		ID->recognizeFeature(net_face, net_so, net_quochuy);

		clock_t end = clock();
		cout << "Time processed is: " << end - start << endl;
		//ID->show_img();
	}

	system("pause");
	return 0;


}
catch (std::exception& e)
{

	cout << e.what() << endl;
	system("pause");
}
