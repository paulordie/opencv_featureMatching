#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"


using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
const char* keys =
        "{ help h |                  | Print help message. }"
        "{ input1 | box.png          | Path to input image 1. }"
        "{ input2 | box_in_scene.png | Path to input image 2. }";
int main( int argc, char* argv[] )
{

    //    cv::Rect roiDtu(11,179,280,160);
    cv::Rect roiDtu(60,260,745,415);
//    cv::Rect roiDtu_bench(343,179,280,160);
    cv::Rect roiDtu_bench(1100,260,730,415);

    //CommandLineParser parser( argc, argv, keys );
    // Mat img_object = imread( samples::findFile( parser.get<String>("input1") ), IMREAD_GRAYSCALE );
    // Mat img_scene = imread( samples::findFile( parser.get<String>("input2") ), IMREAD_GRAYSCALE );
    Mat img_object = imread("/home/pcorrea/Documents/projetos/TPV-OrionPax/featureMatching/saved1.jpg");
    Mat img_scene = imread("/home/pcorrea/Documents/projetos/TPV-OrionPax/featureMatching/saved.jpg");
    if ( img_object.empty() || img_scene.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        //parser.printMessage();
        return -1;
    }
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
//    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_SL2);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Localize the object
    std::vector<Point2f> obj(4);
    std::vector<Point2f> scene(4);
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC );

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)img_object.cols, 0 );
    obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
    obj_corners[3] = Point2f( 0, (float)img_object.rows );
    std::vector<Point2f> scene_corners(4);

    if (!H.empty()) {

        perspectiveTransform( obj_corners, scene_corners, H);
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
              scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 255), 4 );
        line( img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
              scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
              scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar( 255, 255, 0), 4 );
        line( img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
              scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
        //-- Show detected matches
        imshow("Found Blue or Black", img_matches );
        waitKey();
    }
    else {
        std::cout << "Não contém objetos" << std::endl;
    }

    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif