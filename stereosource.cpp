#include "stereosource.h"

StereoSource::StereoSource() {}
StereoSource::~StereoSource() { close(); }

StereoHardware StereoSource::hardware() const { return config_.hardware; }
StereoSourceConfig StereoSource::config() const { return config_; }
bool StereoSource::isOpened() const { return opened_; }
bool StereoSource::hasHardwareDepth() const { return hardwareDepthAvailable_; }
cv::Mat StereoSource::getHardwareDepth() const { return hardwareDepth_; }

bool StereoSource::open(const StereoSourceConfig& config)
{
    close();
    config_ = config;
    hardwareDepthAvailable_ = false;

    switch (config.hardware) {
    case StereoHardware::DualUSB:
        opened_ = openDualUSB();
        break;
    case StereoHardware::DualRTSP:
        opened_ = openDualRTSP();
        break;
    case StereoHardware::RealSense:
    case StereoHardware::ZED:
        opened_ = false;
        break;
    default:
        opened_ = false;
        break;
    }
    return opened_;
}

void StereoSource::close()
{
    if (capL_.isOpened()) capL_.release();
    if (capR_.isOpened()) capR_.release();
    opened_ = false;
    hardwareDepthAvailable_ = false;
}

bool StereoSource::openDualUSB()
{
    if (!capL_.open(config_.leftCameraIndex)) return false;
    if (!capR_.open(config_.rightCameraIndex)) {
        capL_.release();
        return false;
    }

    capL_.set(cv::CAP_PROP_FRAME_WIDTH, config_.targetWidth);
    capL_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.targetHeight);
    capR_.set(cv::CAP_PROP_FRAME_WIDTH, config_.targetWidth);
    capR_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.targetHeight);
    return true;
}

bool StereoSource::openDualRTSP()
{
    if (config_.leftRTSPUrl.empty() || config_.rightRTSPUrl.empty()) return false;

    if (!capL_.open(config_.leftRTSPUrl)) return false;
    if (!capR_.open(config_.rightRTSPUrl)) {
        capL_.release();
        return false;
    }

    capL_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    capR_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    return true;
}

bool StereoSource::grab(cv::Mat& left, cv::Mat& right)
{
    if (!opened_) return false;

    for (int retry = 0; retry < 3; retry++) {
        capL_ >> left;
        capR_ >> right;
        if (!left.empty() && !right.empty()) return true;
    }
    return false;
}
