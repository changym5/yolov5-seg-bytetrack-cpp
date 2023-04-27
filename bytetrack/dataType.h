/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-21 05:49:06
*/
#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;

// Kalmanfilter
// typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
typedef std::pair<KAL_MEAN, KAL_COVA> KAL_DATA;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

struct SegData
{
	SegData() : id(-1), confidence(0.f) {}
	SegData(int _id, float _confidence, const cv::Rect &_box, const cv::Mat &_boxMask)
		: id(_id), confidence(_confidence), box(_box), boxMask(_boxMask) {}
	
	int id;
	float confidence;
	cv::Rect box;
	cv::Mat boxMask;
};
