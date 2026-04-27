#ifndef INFERENCETHREAD_H
#define INFERENCETHREAD_H

#include <QThread>
#include <QImage>
#include <opencv2/opencv.hpp>
#include "yolodetector.h"
#include <atomic>

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

signals:
    void frameReady(const QImage& image, int detectionCount, float fps);
    void inputLost(const QString& msg);

protected:
    void run() override;

private:
    cv::VideoCapture cap_;
    cv::VideoWriter writer_;
    YOLODetector detector_;

    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> isVideo_{false};
    std::atomic<bool> recording_{false};

    cv::Mat currentFrame_;

    void drawDetections(cv::Mat& frame, const std::vector<Detection>& dets);
    static cv::Scalar classColor(int classId);
};

#endif // INFERENCETHREAD_H
