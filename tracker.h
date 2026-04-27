#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "yolodetector.h"

struct Track {
    Detection det;
    int trackId;
};

class Tracker
{
public:
    Tracker(int maxAge = 30, int minHits = 3, float iouThreshold = 0.3f);

    std::vector<Track> update(const std::vector<Detection>& detections);
    void reset();

private:
    struct InternalTrack {
        cv::KalmanFilter kf;
        int trackId;
        int hits;
        int missedFrames;
        cv::Rect lastBbox;
    };

    std::vector<InternalTrack> tracks_;
    int nextId_ = 0;
    int maxAge_;
    int minHits_;
    float iouThreshold_;

    static float computeIoU(const cv::Rect& a, const cv::Rect& b);
    static std::vector<int> hungarian(const cv::Mat& costMatrix, float iouThresh);
    cv::KalmanFilter createKalmanFilter(const cv::Rect& bbox);
    static cv::Rect stateToBbox(const cv::Mat& state);
    static cv::Mat bboxToMeasurement(const cv::Rect& bbox);
};

#endif // TRACKER_H
