#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <time.h>

#include "yolo/yolov5_seg_onnx.h"
#include "bytetrack/ByteTrack.h"

using namespace std;
using namespace cv;
using namespace dnn;

void run_track(cv::Mat &frame, std::vector<Detection> &results, ByteTrack &tracker)
{
    std::vector<STrack> output_stracks = tracker.update(results);

    for (unsigned long i = 0; i < output_stracks.size(); i++)
    {
        std::vector<float> tlwh = output_stracks[i].tlwh;
        cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
        cv::putText(frame, cv::format("%d %d %.2f", output_stracks[i].track_id, output_stracks[i].cls_id, output_stracks[i].score), cv::Point(tlwh[0], tlwh[1] - 5), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
    }
}

// Load images from kitti dataset
void LoadImages(const std::string &strPathToSequence, std::vector<std::string> &vstrImageLeft,
                std::vector<std::string> &vstrImageRight, std::vector<double> &vTimestamps)
{
    std::ifstream fTimes;
    std::string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof())
    {
        std::string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            std::stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    std::string strPrefixLeft = strPathToSequence + "/image_0/";
    std::string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for (int i = 0; i < nTimes; i++)
    {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}

int main()
{
    // tracker parameters
    const int nn_budget = 100;
    const float max_cosine_distance = 0.2;

    const std::string kitti_path = "/root/08";

    // load images from kitti dataset
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(kitti_path, vstrImageLeft, vstrImageRight, vTimestamps);

    const int n_images = vstrImageLeft.size();

    // init detector and tracker
    string model_path = "../models/yolov5m-seg.onnx";
    YoloSegOnnx yolo;
    ByteTrack tracker(max_cosine_distance, nn_budget);

    // Net net;
    if (yolo.ReadModel(model_path, true, 0, true))
    {
        cout << "read net ok!" << endl;
    }
    else
    {
        return -1;
    }
    // random color
    vector<Scalar> color;
    srand(time(0));
    for (int i = 0; i < 80; i++)
    {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(Scalar(b, g, r));
    }

    for (int i = 0; i < n_images; i++)
    {
        cv::Mat frame = cv::imread(vstrImageLeft[i]);
        cv::Mat frame_show = frame.clone();

        vector<OutputSeg> result;
        yolo.OnnxDetect(frame, result);

        vector<Detection> detections;
        for (auto &&seg : result)
        {
            Detection detect;
            detect.box = seg.box;
            detect.classId = seg.id;
            detect.conf = seg.confidence;
            detections.push_back(detect);
        }

        run_track(frame_show, detections, tracker);  // run tracker and draw tracking and box

        DrawPred(frame_show, result, yolo._className, color);  // draw mask

        cv::imshow("img", frame_show);

        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}
