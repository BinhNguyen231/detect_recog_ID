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


#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace dlib;

// The first thing we do is define our CNN.  The CNN is going to be evaluated
// convolutionally over an entire image pyramid.  Think of it like a normal
// sliding window classifier.  This means you need to define a CNN that can look
// at some part of an image and decide if it is an object of interest.  In this
// example I've defined a CNN with a receptive field of a little over 50x50
// pixels.  This is reasonable for face detection since you can clearly tell if
// a 50x50 image contains a face.  Other applications may benefit from CNNs with
// different architectures.  
// 
// In this example our CNN begins with 3 downsampling layers.  These layers will
// reduce the size of the image by 8x and output a feature map with
// 32 dimensions.  Then we will pass that through 4 more convolutional layers to
// get the final output of the network.  The last layer has only 1 channel and
// the values in that last channel are large when the network thinks it has
// found an object at a particular location.


// Let's begin the network definition by creating some network blocks.

// A 5x5 conv layer that does 2x downsampling
template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
// A 3x3 conv layer that doesn't do any downsampling
template <long num_filters, typename SUBNET> using con3 = con<num_filters, 3, 3, 1, 1, SUBNET>;

// Now we can define the 8x downsampling block in terms of conv5d blocks.  We
// also use relu and batch normalization in the standard way.
template <typename SUBNET> using downsampler = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32, SUBNET>>>>>>>>>;

// The rest of the network will be 3x3 conv layers with batch normalization and
// relu.  So we define the 3x3 block we will use here.
template <typename SUBNET> using rcon3 = relu<bn_con<con3<32, SUBNET>>>;

// Finally, we define the entire network.   The special input_rgb_image_pyramid
// layer causes the network to operate over a spatial pyramid, making the detector
// scale invariant.  
using net_type = loss_mmod<con<1, 6, 6, 1, 1, rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        return 0;
    }
    const std::string data_directory = argv[1];

    std::vector<matrix<rgb_pixel>> images_train, images_test;
    std::vector<std::vector<mmod_rect>> boxes_train, boxes_test;
    load_image_dataset(images_train, boxes_train, data_directory + "/data.xml");
    load_image_dataset(images_test, boxes_test, data_directory + "/data.xml");


    cout << "num training images: " << images_train.size() << endl;
    cout << "num testing images:  " << images_test.size() << endl;


    mmod_options options(boxes_train, 30, 30);
    
    // Now we are ready to create our network and trainer.  
    net_type net(options);
    net_type test(options);
 
    // The MMOD loss requires that the number of filters in the final network layer equal
    // options.detector_windows.size().  So we set that here as well.
    net.subnet().layer_details().set_num_filters(options.detector_windows.size());
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file("mmod_sync", std::chrono::minutes(5));
    trainer.set_iterations_without_progress_threshold(300);


    std::vector<matrix<rgb_pixel>> mini_batch_samples;
    std::vector<std::vector<mmod_rect>> mini_batch_labels;
    random_cropper cropper;
    cropper.set_chip_dims(100, 100);
    cropper.set_min_object_size(30, 30);
    cropper.set_max_rotation_degrees(360);
    cropper.set_randomly_flip(false);
    dlib::rand rnd;
    
    while (trainer.get_learning_rate() >= 1e-4)
    {
        cropper(200, images_train, boxes_train, mini_batch_samples, mini_batch_labels);
        // We can also randomly jitter the colors and that often helps a detector
        // generalize better to new images.
        for (auto&& img : mini_batch_samples)
            disturb_colors(img, rnd);

        trainer.train_one_step(mini_batch_samples, mini_batch_labels);
    }
    // wait for training threads to stop
    trainer.get_net();
    cout << "done training" << endl;

    // Save the network to disk
    net.clean();
    serialize("mmod_network.dat") << net;

    deserialize("mmod_network.dat") >> test;
    cout << "training results: " << test_object_detection_function(test, images_train, boxes_train) << endl;
    cout << "testing results:  " << test_object_detection_function(test, images_test, boxes_test) << endl;

    // Now lets run the detector on the testing images and look at the outputs.  
    image_window win;
    for (auto&& img : images_test)
    {
        pyramid_up(img);
        auto dets = test(img);
        win.clear_overlay();
        win.set_image(img);
        for (auto&& d : dets)
        {
            win.add_overlay(d);
        }
        cin.get();
    }
    return 0;
}
catch (std::exception & e)
{
    cout << e.what() << endl;
}




