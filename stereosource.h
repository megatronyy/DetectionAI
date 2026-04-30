#ifndef STEREOSOURCE_H
#define STEREOSOURCE_H

#include <opencv2/opencv.hpp>
#include <string>

enum class StereoHardware {
    SingleMono,
    DualUSB,
    DualRTSP,
    RealSense,
    ZED
};

struct StereoSourceConfig {
    StereoHardware hardware = StereoHardware::SingleMono;
    int leftCameraIndex = 0;
    int rightCameraIndex = 1;
    std::string leftRTSPUrl;
    std::string rightRTSPUrl;
    int targetWidth = 640;
    int targetHeight = 480;
};

class StereoSource
{
public:
    StereoSource();
    ~StereoSource();

    bool open(const StereoSourceConfig& config);
    void close();
    bool isOpened() const;

    bool grab(cv::Mat& left, cv::Mat& right);

    StereoHardware hardware() const;
    StereoSourceConfig config() const;

    bool hasHardwareDepth() const;
    cv::Mat getHardwareDepth() const;

private:
    StereoSourceConfig config_;
    cv::VideoCapture capL_, capR_;
    bool opened_ = false;
    bool hardwareDepthAvailable_ = false;
    cv::Mat hardwareDepth_;

    bool openDualUSB();
    bool openDualRTSP();
};

#endif // STEREOSOURCE_H
