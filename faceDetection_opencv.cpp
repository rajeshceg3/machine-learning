#include <iostream>
#include <stdio.h>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// Library dependencies for linking -lopencv_core -lopencv_objdetect -lopencv_highgui -lopencv_videoio -lopencv_imgproc

using std::vector;
using namespace cv;

int main( int argc, const char** argv )
{   
    // Load classifier and input image from cmd line arguments
    CascadeClassifier face_detector;
    face_detector.load(argv[1]);
    Mat rgb_img = imread(argv[2]);
	
	  // Covert to grayscale and equalize histogram
    Mat gray_img;
    cvtColor( rgb_img, gray_img, COLOR_BGR2GRAY );
    equalizeHist( gray_img, gray_img );
    
    // Identify face using detectMultiScale method
	  std::vector<Rect> faces;
    face_detector.detectMultiScale( gray_img, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    for ( size_t i = 0; i < faces.size(); i++ )
        rectangle(rgb_img, faces[i], Scalar(0,255,0));

    imshow( "Face identified", rgb_img );
    waitKey();
    return 0;
}
