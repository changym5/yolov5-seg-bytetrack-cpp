#pragma once
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "../bytetrack/dataType.h"

#define YOLO_P6 false
#define ORT_OLD_VISON 12

struct MaskParams
{
	int segChannels = 32;
	int segWidth = 160;
	int segHeight = 160;
	int netWidth = 640;
	int netHeight = 640;
	float maskThreshold = 0.5;
	cv::Size srcImgShape;
	cv::Vec4d params;
};
bool CheckParams(int netHeight, int netWidth, const int *netStride, int strideSize);
void DrawPred(cv::Mat &img, std::vector<SegData> result, std::vector<std::string> classNames, std::vector<cv::Scalar> color);
void LetterBox(const cv::Mat &image, cv::Mat &outImage,
			   cv::Vec4d &params, //[ratio_x,ratio_y,dw,dh]
			   const cv::Size &newShape = cv::Size(640, 640),
			   bool autoShape = false,
			   bool scaleFill = false,
			   bool scaleUp = true,
			   int stride = 32,
			   const cv::Scalar &color = cv::Scalar(114, 114, 114));
void GetMask(const cv::Mat &maskProposals, const cv::Mat &maskProtos, std::vector<SegData> &output, const MaskParams &maskParams);
void GetMask2(const cv::Mat &maskProposals, const cv::Mat &maskProtos, SegData &output, const MaskParams &maskParams);
