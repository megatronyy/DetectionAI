#ifndef STEREOMATCHER_H
#define STEREOMATCHER_H

#include <opencv2/opencv.hpp>

struct SGBMParams {
    int blockSize = 5;
    int minDisparity = 0;
    int numDisparities = 64;
    int uniquenessRatio = 10;
    int speckleWindowSize = 100;
    int speckleRange = 32;
    int disp12MaxDiff = 1;
    int preFilterCap = 63;
    int mode = cv::StereoSGBM::MODE_SGBM_3WAY;

    float baselineMeters = 0.06f;
    float focalLengthPixels = 700.f;
};

struct DepthResult {
    float distance;
    float confidence;
};

class StereoMatcher
{
public:
    StereoMatcher();

    void setParams(const SGBMParams& params);
    SGBMParams params() const;

    cv::Mat computeDisparity(const cv::Mat& leftRect, const cv::Mat& rightRect);
    DepthResult computeDistance(const cv::Mat& disparityMap, const cv::Rect& bbox) const;
    cv::Mat disparityToDepth(const cv::Mat& disparity) const;
    cv::Mat disparityColormap(const cv::Mat& disparity) const;

private:
    SGBMParams params_;
    cv::Ptr<cv::StereoSGBM> sgbm_;
    void createSGBM();
};

#endif // STEREOMATCHER_H
