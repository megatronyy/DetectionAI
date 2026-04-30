#include "stereomatcher.h"
#include <algorithm>
#include <cmath>
#include <numeric>

StereoMatcher::StereoMatcher()
{
    createSGBM();
}

void StereoMatcher::setParams(const SGBMParams& params)
{
    params_ = params;
    createSGBM();
}

SGBMParams StereoMatcher::params() const { return params_; }

void StereoMatcher::createSGBM()
{
    sgbm_ = cv::StereoSGBM::create(
        params_.minDisparity, params_.numDisparities, params_.blockSize,
        params_.preFilterCap, params_.uniquenessRatio,
        params_.speckleWindowSize, params_.speckleRange,
        params_.disp12MaxDiff, params_.mode);
}

cv::Mat StereoMatcher::computeDisparity(const cv::Mat& leftRect, const cv::Mat& rightRect)
{
    cv::Mat disp;
    sgbm_->compute(leftRect, rightRect, disp);
    return disp;
}

DepthResult StereoMatcher::computeDistance(const cv::Mat& disparityMap, const cv::Rect& bbox) const
{
    DepthResult result;
    result.distance = -1.f;
    result.confidence = 0.f;

    if (disparityMap.empty() || params_.baselineMeters <= 0 || params_.focalLengthPixels <= 0)
        return result;

    cv::Rect roi = bbox & cv::Rect(0, 0, disparityMap.cols, disparityMap.rows);
    if (roi.area() < 4) return result;

    cv::Mat roiDisp = disparityMap(roi);
    std::vector<float> validDisps;
    validDisps.reserve(roi.area());

    for (int y = 0; y < roiDisp.rows; y++) {
        const auto* row = roiDisp.ptr<int16_t>(y);
        for (int x = 0; x < roiDisp.cols; x++) {
            float d = row[x] / 16.0f;
            if (d > 1.f) validDisps.push_back(d);
        }
    }

    if (validDisps.empty()) return result;

    size_t n = validDisps.size();
    std::nth_element(validDisps.begin(), validDisps.begin() + n / 2, validDisps.end());
    float medianDisp = validDisps[n / 2];

    result.distance = (params_.baselineMeters * params_.focalLengthPixels) / medianDisp;
    result.confidence = (float)n / roi.area();
    return result;
}

cv::Mat StereoMatcher::disparityToDepth(const cv::Mat& disparity) const
{
    cv::Mat depth;
    disparity.convertTo(depth, CV_32F, 1.0 / 16.0);
    cv::Mat valid = (depth > 1.f);
    depth.setTo(0, ~valid);
    cv::divide(params_.baselineMeters * params_.focalLengthPixels, depth, depth);
    depth.setTo(0, ~valid);
    return depth;
}

cv::Mat StereoMatcher::disparityColormap(const cv::Mat& disparity) const
{
    cv::Mat disp8;
    cv::normalize(disparity, disp8, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Mat mask = (disparity > 0);
    disp8.setTo(0, ~mask);
    cv::Mat colorDisp;
    cv::applyColorMap(disp8, colorDisp, cv::COLORMAP_JET);
    colorDisp.setTo(cv::Scalar(0, 0, 0), ~mask);
    return colorDisp;
}
