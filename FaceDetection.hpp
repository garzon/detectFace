#ifndef FACEDETECTION_HPP
#define FACEDETECTION_HPP
/*Copyright (c) 2014, School of Computer Science, Fudan University*/

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <vector>
#include <caffe/caffe.hpp>
#include <caffe.pb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "system_struct.hpp"

class FaceDetection{

    const options &_ops;
    caffe::Net<float> *net;

    cv::HOGDescriptor hog;

    shared_data::bbox facetmp;
    cv::Mat ROI,inputImg;
    float *blobData,*dummyData;
    std::vector<cv::Rect> tmp;

    void preprocessing(cv::Mat &img){
        cv::resize(img,img,cv::Size(_ops.face_d.width,_ops.face_d.height));
        cv::cvtColor(img,img,CV_BGR2GRAY);
        cv::equalizeHist(img,img);
    }

    template <typename T>
    void iterAll(cv::Mat &a,const std::function<void (T&)> &f){
        for(auto b=a.begin<T>();b!=a.end<T>();b++){
            f(*b);
        }
    }

public:

    FaceDetection(const options &ops):
        _ops(ops),
        hog(cv::Size(48,48), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9)
    {

        hog.load(_ops.face_d.hog_head);

        cudaSetDevice(0);
        caffe::Caffe::set_phase(caffe::Caffe::TEST);
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
        caffe::NetParameter test_net_param;
        caffe::ReadProtoFromTextFile(ops.face_d.netParams, &test_net_param);
        net=new caffe::Net<float>(test_net_param);
        caffe::NetParameter trained_net_param;
        caffe::ReadProtoFromBinaryFile(ops.face_d.netWeights, &trained_net_param);
        net->CopyTrainedLayersFrom(trained_net_param);

        blobData=new float[ops.face_d.height*ops.face_d.width];
        dummyData=new float[1];

        caffe::MemoryDataLayer<float> *input_layer=(caffe::MemoryDataLayer<float> *)net->layers()[0].get();
        input_layer->Reset(blobData,dummyData,1);

    }

    bool isFace(const cv::Mat &window){
        // return true;
        inputImg=window.clone();
        preprocessing(inputImg);
        float *p=blobData;
        iterAll<uchar>(inputImg,[&p](uchar &v){
            *(p++)=static_cast<float>(v*1.0/256);
        });

        const float* result = net->ForwardPrefilled()[1]->cpu_data();
        std::cout<<"Prob: "<<result[1]<<std::endl;
        return (result[1] >= 0.04);
    }

    void hogSearch(const cv::Mat &ROI,std::vector<cv::Rect> &result) const {
        result.clear();

        //for(int i=0;i<ROI.rows-111;i+=20){
        //    for(int j=0;j<ROI.cols-111;j+=20){
        //        result.push_back(cv::Rect(j,i,100,100));
        //    }
        //}

        //return;

        if(ROI.rows>=48)
            if(ROI.cols>=48)
                hog.detectMultiScale(ROI, result, 1, cv::Size(8,8), cv::Size(0,0), 1.1, 3, false);
        for(auto &face: result){
            if(face.height*1.3+face.y<ROI.rows)
                face.height=floor(face.height*1.3);
            //if(face.width*1.3+face.y<ROI.cols)
            //    face.width=floor(face.width*1.3);
        }
    }

    void processFrame(shared_data &data){
        for(const auto &bbox: data.im_data.im_ROI){
            ROI=data.im_data.image(bbox);
            hogSearch(ROI,tmp);
            for(cv::Rect &window: tmp){
                if(isFace(ROI(window))){
                    window.x+=bbox.x;
                    window.y+=bbox.y;
                    facetmp=shared_data::bbox(window);
                    facetmp.type_label=TYPE_FACE;                              // means that it's a face
                    data.im_boxes.push_back(facetmp);
                };
            }
        }
    }

    ~FaceDetection(){
        delete net;
        delete [] dummyData;
        delete [] blobData;
    }
};

#endif // FACEDETECTION_HPP
