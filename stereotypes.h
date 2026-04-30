#ifndef STEREOTYPES_H
#define STEREOTYPES_H

#include <opencv2/opencv.hpp>
#include <string>

struct StereoCalibration {
    cv::Mat cameraMatrixL, distCoeffsL;
    cv::Mat cameraMatrixR, distCoeffsR;
    cv::Mat R, T;
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validRoiL, validRoiR;
    bool valid = false;
    double reprojectionError = -1.0;
};

class StereoRectifier
{
public:
    StereoRectifier();

    bool loadCalibration(const std::string& path);
    bool saveCalibration(const std::string& path) const;
    void setCalibration(const StereoCalibration& cal);
    bool isCalibrated() const;
    const StereoCalibration& calibration() const;

    void rectify(const cv::Mat& leftRaw, const cv::Mat& rightRaw,
                 cv::Mat& leftRect, cv::Mat& rightRect);

    void initRectifyMaps(int imageWidth, int imageHeight);

private:
    StereoCalibration cal_;
    cv::Mat mapL1_, mapL2_;
    cv::Mat mapR1_, mapR2_;
    int lastWidth_ = 0, lastHeight_ = 0;
};

#endif // STEREOTYPES_H
