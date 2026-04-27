#ifndef INFERENCETHREAD_H
#define INFERENCETHREAD_H

#include <QThread>
#include <QImage>
#include <QMap>
#include <opencv2/opencv.hpp>
#include "yolodetector.h"
#include "tracker.h"
#include <atomic>
#include <mutex>

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

    void setLoopEnabled(bool enabled);
    bool isLoopEnabled() const;
    QSize frameSize() const;
    std::vector<Detection> lastDetections() const;

signals:
    void frameReady(const QImage& image, int detectionCount, float fps,
                    float inferMs, const QMap<int,int>& classCounts);
    void inputLost(const QString& msg);

protected:
    void run() override;

private:
    cv::VideoCapture cap_;
    cv::VideoWriter writer_;
    YOLODetector detector_;
    Tracker tracker_;

    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> isVideo_{false};
    std::atomic<bool> recording_{false};
    std::atomic<bool> trackingEnabled_{false};
    std::atomic<bool> loopEnabled_{false};

    cv::Mat currentFrame_;
    QSize lastFrameSize_;
    std::vector<Detection> lastDetections_;
    mutable std::mutex detectionsMutex_;

    void drawDetections(cv::Mat& frame, const std::vector<Detection>& dets);
    void drawTracks(cv::Mat& frame, const std::vector<Track>& tracks);
    static cv::Scalar classColor(int classId);
};

#endif // INFERENCETHREAD_H
