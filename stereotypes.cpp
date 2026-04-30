#include "stereotypes.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

StereoRectifier::StereoRectifier() {}

bool StereoRectifier::isCalibrated() const { return cal_.valid; }
const StereoCalibration& StereoRectifier::calibration() const { return cal_; }

void StereoRectifier::setCalibration(const StereoCalibration& cal)
{
    cal_ = cal;
    lastWidth_ = 0;
    lastHeight_ = 0;
}

void StereoRectifier::initRectifyMaps(int imageWidth, int imageHeight)
{
    if (!cal_.valid) return;
    if (imageWidth == lastWidth_ && imageHeight == lastHeight_) return;

    cv::initUndistortRectifyMap(cal_.cameraMatrixL, cal_.distCoeffsL,
        cal_.R1, cal_.P1, cv::Size(imageWidth, imageHeight),
        CV_32FC1, mapL1_, mapL2_);
    cv::initUndistortRectifyMap(cal_.cameraMatrixR, cal_.distCoeffsR,
        cal_.R2, cal_.P2, cv::Size(imageWidth, imageHeight),
        CV_32FC1, mapR1_, mapR2_);

    lastWidth_ = imageWidth;
    lastHeight_ = imageHeight;
}

void StereoRectifier::rectify(const cv::Mat& leftRaw, const cv::Mat& rightRaw,
                               cv::Mat& leftRect, cv::Mat& rightRect)
{
    if (!cal_.valid) {
        leftRaw.copyTo(leftRect);
        rightRaw.copyTo(rightRect);
        return;
    }

    initRectifyMaps(leftRaw.cols, leftRaw.rows);
    cv::remap(leftRaw, leftRect, mapL1_, mapL2_, cv::INTER_LINEAR);
    cv::remap(rightRaw, rightRect, mapR1_, mapR2_, cv::INTER_LINEAR);
}

bool StereoRectifier::saveCalibration(const std::string& path) const
{
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;

    fs << "cameraMatrixL" << cal_.cameraMatrixL;
    fs << "distCoeffsL" << cal_.distCoeffsL;
    fs << "cameraMatrixR" << cal_.cameraMatrixR;
    fs << "distCoeffsR" << cal_.distCoeffsR;
    fs << "R" << cal_.R;
    fs << "T" << cal_.T;
    fs << "R1" << cal_.R1;
    fs << "R2" << cal_.R2;
    fs << "P1" << cal_.P1;
    fs << "P2" << cal_.P2;
    fs << "Q" << cal_.Q;
    fs << "reprojectionError" << cal_.reprojectionError;
    fs.release();
    return true;
}

bool StereoRectifier::loadCalibration(const std::string& path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    fs["cameraMatrixL"] >> cal_.cameraMatrixL;
    fs["distCoeffsL"] >> cal_.distCoeffsL;
    fs["cameraMatrixR"] >> cal_.cameraMatrixR;
    fs["distCoeffsR"] >> cal_.distCoeffsR;
    fs["R"] >> cal_.R;
    fs["T"] >> cal_.T;
    fs["R1"] >> cal_.R1;
    fs["R2"] >> cal_.R2;
    fs["P1"] >> cal_.P1;
    fs["P2"] >> cal_.P2;
    fs["Q"] >> cal_.Q;

    if (!cal_.R1.empty() && !cal_.P1.empty() && !cal_.Q.empty()) {
        cal_.valid = true;
        cal_.reprojectionError = -1.0;
        if (!fs["reprojectionError"].empty())
            fs["reprojectionError"] >> cal_.reprojectionError;
        lastWidth_ = 0;
        lastHeight_ = 0;
    }

    fs.release();
    return cal_.valid;
}
