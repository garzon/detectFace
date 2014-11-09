#include <string>

#include <opencv2/highgui/highgui.hpp>

#include "system_struct.hpp"
#include "FaceDetection.hpp"

using namespace cv;
using namespace std;

int main(){

    const string fileName="/home/garzon/Downloads/facesImages/7.jpg";

    options ops;
    FaceDetection FD(ops);

    shared_data sd(imread(fileName));
    sd.im_data.im_ROI.push_back(Rect(0,0,sd.im_data.image.cols,sd.im_data.image.rows));

    FD.processFrame(sd);

    for(auto &a: sd.im_boxes){
        rectangle(sd.im_data.image,a,Scalar(255,0,0));
    }

    imshow("res",sd.im_data.image);
    waitKey();

    return 0;
}
