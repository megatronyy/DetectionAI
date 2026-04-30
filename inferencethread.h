#ifndef INFERENCETHREAD_H
#define INFERENCETHREAD_H

#include <QThread>
#include <QImage>
#include <QMap>
#include <opencv2/opencv.hpp>
#include "yolodetector.h"
#include "tracker.h"
#include "stereosource.h"
#include "stereotypes.h"
#include "stereomatcher.h"
#include <atomic>
#include <mutex>

struct TrackRecord {
    int trackId;
    int classId;
    int64_t timestampMs;
    int x, y, width, height;
    float speed, angle;
    float distance;
    std::vector<float> kpData;
};

class InferenceThread : public QThread
{
    Q_OBJECT
public:
    InferenceThread(QObject* parent = nullptr);
    ~InferenceThread();

    bool openCamera(int index);
    bool openVideo(const std::string& path);
    void setPaused(bool paused);
    void stop();

    YOLODetector& detector();
    bool isVideoSource() const;

    void startRecording(const std::string& path, double fps, int width, int height);
    void stopRecording();
    bool isRecording() const;

    void setTrackingEnabled(bool enabled);
    bool isTrackingEnabled() const;
    void resetTracker();
    void resetTrackCounts();

    void setTrajectoryEnabled(bool enabled);
    bool isTrajectoryEnabled() const;

    void setSpeedEnabled(bool enabled);
    bool isSpeedEnabled() const;

    void setSkeletonEnabled(bool enabled);
    bool isSkeletonEnabled() const;
    void setKeypointConfThreshold(float t);
    float keypointConfThreshold() const;

    bool openStereo(const StereoSourceConfig& config);
    void setStereoMode(bool enabled);
    bool isStereoMode() const;
    void setDepthOverlay(bool enabled);
    bool depthOverlayEnabled() const;
    void setStereoCalibration(const StereoCalibration& cal);
    void setSGBMParams(const SGBMParams& params);
    SGBMParams sgbmParams() const;
    StereoSourceConfig stereoSourceConfig() const;
    StereoSource& stereoSource();

    void setCountingLine(const CountingLine& line);
    void clearCountingLine();
    bool hasCountingLine() const;
    QMap<int, QMap<int, int>> crossingCountsByDir() const;
    void resetCrossingCounts();

    void setLoopEnabled(bool enabled);
    bool isLoopEnabled() const;
    QSize frameSize() const;
    std::vector<Detection> lastDetections() const;

    std::vector<TrackRecord> trackHistory() const;
    void clearTrackHistory();

signals:
    void frameReady(const QImage& image, int detectionCount, float fps,
                    float inferMs, const QMap<int,int>& classCounts);
    void inputLost(const QString& msg);
    void trackingStatsUpdated(const QMap<int,int>& uniqueCounts, int totalUnique);
    void crossingStatsUpdated(const QMap<int, QMap<int, int>>& counts);
    void poseDataUpdated(const std::vector<Detection>& dets);
    void depthMapReady(const QImage& depthViz, float avgDepth);

protected:
    void run() override;

private:
    cv::VideoCapture cap_;
    cv::VideoWriter writer_;
    YOLODetector detector_;
    Tracker tracker_;
    StereoSource stereoSource_;
    StereoRectifier rectifier_;
    StereoMatcher matcher_;
    cv::Mat rightFrame_;

    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> isVideo_{false};
    std::atomic<bool> recording_{false};
    std::atomic<bool> trackingEnabled_{false};
    std::atomic<bool> trajectoryEnabled_{false};
    std::atomic<bool> speedEnabled_{false};
    std::atomic<bool> skeletonEnabled_{true};
    std::atomic<bool> stereoMode_{false};
    std::atomic<bool> depthOverlayEnabled_{false};
    std::atomic<bool> loopEnabled_{false};
    float keypointConfThreshold_ = 0.5f;

    cv::Mat currentFrame_;
    QSize lastFrameSize_;
    std::vector<Detection> lastDetections_;
    mutable std::mutex detectionsMutex_;
    std::vector<TrackRecord> trackHistory_;
    mutable std::mutex historyMutex_;
    static const int MAX_HISTORY = 500000;

    void drawDetections(cv::Mat& frame, const std::vector<Detection>& dets);
    void drawTracks(cv::Mat& frame, const std::vector<Track>& tracks);
    void drawTrajectory(cv::Mat& frame, const std::vector<Track>& tracks);
    void drawSkeleton(cv::Mat& frame, const Detection& det);
    void drawSkeletons(cv::Mat& frame, const std::vector<Detection>& dets);
    void drawPoseTracks(cv::Mat& frame, const std::vector<Track>& tracks);
    void drawCountingLine(cv::Mat& frame);
    static void drawLabel(cv::Mat& frame, const cv::Rect& bbox, int classId, const std::string& label);
    static cv::Scalar classColor(int classId);
};

#endif // INFERENCETHREAD_H
