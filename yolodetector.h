#ifndef YOLODETECTOR_H
#define YOLODETECTOR_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <array>
#include <utility>
#include <QSet>
#include <mutex>

struct Keypoint {
    cv::Point2f pt;
    float confidence;
};

struct Detection {
    int classId;
    float confidence;
    cv::Rect bbox;
    std::vector<Keypoint> keypoints;
    float distance = -1.f;
};

enum class ModelType { Detection, Pose };

class YOLODetector
{
public:
    YOLODetector();

    bool loadModel(const std::wstring& modelPath, int threads = 4);
    std::vector<Detection> detect(const cv::Mat& frame);

    void setConfThreshold(float t);
    float confThreshold() const;
    void setIouThreshold(float t);
    float iouThreshold() const;
    void setEnabledClasses(const QSet<int>& classes);
    QSet<int> enabledClasses() const;
    bool isGpuEnabled() const;
    bool isLoaded() const;
    ModelType modelType() const;
    bool isPoseModel() const;
    int numKeypoints() const;

    static const std::vector<std::string> CLASS_NAMES;
    static const int NUM_CLASSES = 80;

    static const int POSE_CHANNELS = 56;
    static const std::vector<std::string> KEYPOINT_NAMES;
    static const std::vector<std::pair<int,int>> SKELETON_CONNECTIONS;

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> inNamesC_;
    std::vector<const char*> outNamesC_;

    int inputW_ = 640;
    int inputH_ = 640;
    float confThreshold_ = 0.25f;
    float iouThreshold_ = 0.45f;
    bool gpuEnabled_ = false;
    bool loaded_ = false;
    ModelType modelType_ = ModelType::Detection;
    int numKeypoints_ = 0;

    float scaleX_ = 1.f, scaleY_ = 1.f;
    int padX_ = 0, padY_ = 0;

    QSet<int> enabledClasses_;
    mutable std::mutex filterMutex_;

    // Pre-allocated buffers (avoid per-frame heap allocation)
    std::vector<float> inputBuf_;
    cv::Mat letterboxBuf_;
    std::array<int64_t, 4> inputShape_;
    Ort::MemoryInfo memInfo_;

    void letterbox(const cv::Mat& src, cv::Mat& dst);
    void preprocess(const cv::Mat& src);
    std::vector<Detection> postprocess(std::vector<Ort::Value>& outputs, int origW, int origH);

    static float computeIoU(const cv::Rect& a, const cv::Rect& b);
    static std::vector<Detection> nms(std::vector<Detection>& dets, float iouThresh);
};

#endif // YOLODETECTOR_H
