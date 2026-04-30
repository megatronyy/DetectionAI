#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <QMap>
#include <QSet>
#include "yolodetector.h"

struct CountingLine {
    cv::Point pt1, pt2;
    std::string label;
};

struct CrossingEvent {
    int trackId;
    int classId;
    int direction;       // +1 forward, -1 reverse
    int64_t timestampMs;
    float distance;      // distance at crossing moment
};

struct Track {
    Detection det;
    int trackId;
    std::vector<cv::Point> trajectory;
    std::vector<float> distanceHistory;
    float speed = 0.f;
    float angle = 0.f;
    float avgDistance = -1.f;
};

class Tracker
{
public:
    Tracker(int maxAge = 30, int minHits = 3, float iouThreshold = 0.3f);

    std::vector<Track> update(const std::vector<Detection>& detections);
    void reset();
    void resetCounts();

    QMap<int, int> uniqueCounts() const;
    int totalUnique() const;

    void setCountingLine(const CountingLine& line);
    void clearCountingLine();
    bool hasCountingLine() const;
    CountingLine countingLine() const;
    QMap<int, QMap<int, int>> crossingCountsByDir() const;
    void resetCrossingCounts();
    void setCurrentTime(int64_t ms);

private:
    struct InternalTrack {
        cv::KalmanFilter kf;
        int trackId;
        int hits;
        int missedFrames;
        cv::Rect lastBbox;
        int classId;
        float confidence;
        std::vector<cv::Point> trajectory;
        int lastSide = 0;
        std::vector<Keypoint> lastKeypoints;
        std::vector<float> distanceHistory;
    };

    std::vector<InternalTrack> tracks_;
    int nextId_ = 0;
    int maxAge_;
    int minHits_;
    float iouThreshold_;
    QMap<int, QSet<int>> seenIdsByClass_;

    CountingLine countLine_;
    bool hasLine_ = false;
    std::vector<CrossingEvent> crossings_;
    QMap<int, QMap<int, int>> crossingCountsByDir_;
    int64_t currentTimeMs_ = 0;

    static const int MAX_TRAJECTORY_LEN = 200;

    static float computeIoU(const cv::Rect& a, const cv::Rect& b);
    static std::vector<int> hungarian(const cv::Mat& costMatrix, float iouThresh);
    static int sideOfLine(const cv::Point& p, const CountingLine& line);
    cv::KalmanFilter createKalmanFilter(const cv::Rect& bbox);
    static cv::Rect stateToBbox(const cv::Mat& state);
    static cv::Mat bboxToMeasurement(const cv::Rect& bbox);
};

#endif // TRACKER_H
