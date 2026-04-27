#include "yolodetector.h"
#include <algorithm>
#include <cstring>

#ifdef USE_CUDA
#include <cuda_provider_factory.h>
#endif

const std::vector<std::string> YOLODetector::CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

YOLODetector::YOLODetector()
    : env_(ORT_LOGGING_LEVEL_WARNING, "YOLO11")
    , memInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , inputShape_{1, 3, inputH_, inputW_}
{
}

bool YOLODetector::loadModel(const std::wstring& modelPath, int threads)
{
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
        try {
            OrtCUDAProviderOptions cudaOpts;
            opts.AppendExecutionProvider_CUDA(cudaOpts);
            gpuEnabled_ = true;
        } catch (...) {
            gpuEnabled_ = false;
        }
#endif

        session_ = Ort::Session(env_, modelPath.c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < session_.GetInputCount(); i++) {
            auto name = session_.GetInputNameAllocated(i, alloc);
            inputNames_.emplace_back(name.get());
        }
        for (size_t i = 0; i < session_.GetOutputCount(); i++) {
            auto name = session_.GetOutputNameAllocated(i, alloc);
            outputNames_.emplace_back(name.get());
        }

        // Cache const char* pointers (stable as long as strings are not modified)
        inNamesC_.resize(inputNames_.size());
        for (size_t i = 0; i < inputNames_.size(); i++)
            inNamesC_[i] = inputNames_[i].c_str();
        outNamesC_.resize(outputNames_.size());
        for (size_t i = 0; i < outputNames_.size(); i++)
            outNamesC_[i] = outputNames_[i].c_str();

        // Pre-allocate input buffer (3 channels * 640 * 640)
        inputBuf_.resize(3 * inputW_ * inputH_);

        loaded_ = true;
        return true;
    } catch (...) {
        loaded_ = false;
        return false;
    }
}

void YOLODetector::letterbox(const cv::Mat& src, cv::Mat& dst)
{
    float scale = std::min((float)inputW_ / src.cols, (float)inputH_ / src.rows);
    int newW = (int)(src.cols * scale);
    int newH = (int)(src.rows * scale);

    scaleX_ = scale;
    scaleY_ = scale;
    padX_ = (inputW_ - newW) / 2;
    padY_ = (inputH_ - newH) / 2;

    if (dst.size() != cv::Size(inputW_, inputH_) || dst.type() != CV_8UC3)
        dst = cv::Mat(inputH_, inputW_, CV_8UC3, cv::Scalar(114, 114, 114));
    else
        dst.setTo(cv::Scalar(114, 114, 114));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH));
    resized.copyTo(dst(cv::Rect(padX_, padY_, newW, newH)));
}

void YOLODetector::preprocess(const cv::Mat& src)
{
    cv::Mat rgb;
    cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    cv::Mat channels[3];
    cv::split(rgb, channels);

    size_t plane = (size_t)inputW_ * inputH_;
    std::memcpy(inputBuf_.data(), channels[0].data, plane * sizeof(float));
    std::memcpy(inputBuf_.data() + plane, channels[1].data, plane * sizeof(float));
    std::memcpy(inputBuf_.data() + 2 * plane, channels[2].data, plane * sizeof(float));
}

float YOLODetector::computeIoU(const cv::Rect& a, const cv::Rect& b)
{
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    int inter = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.width * a.height + b.width * b.height - inter;
    return unionArea > 0 ? (float)inter / unionArea : 0.f;
}

std::vector<Detection> YOLODetector::nms(std::vector<Detection>& dets, float iouThresh)
{
    std::sort(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> result;
    result.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            if (dets[i].classId == dets[j].classId &&
                computeIoU(dets[i].bbox, dets[j].bbox) > iouThresh) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

std::vector<Detection> YOLODetector::postprocess(
    std::vector<Ort::Value>& outputs, int origW, int origH)
{
    float* data = outputs[0].GetTensorMutableData<float>();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    int dim1 = (int)shape[1], dim2 = (int)shape[2];
    int numBoxes, channels;
    bool transposed = (dim1 < dim2);

    if (transposed) {
        channels = dim1;
        numBoxes = dim2;
    } else {
        numBoxes = dim1;
        channels = dim2;
    }

    std::vector<Detection> dets;
    dets.reserve(numBoxes);

    for (int i = 0; i < numBoxes; i++) {
        float bx, by, bw, bh;
        float conf = 0;
        int cls = 0;

        if (transposed) {
            bx = data[0 * numBoxes + i];
            by = data[1 * numBoxes + i];
            bw = data[2 * numBoxes + i];
            bh = data[3 * numBoxes + i];
            for (int j = 4; j < channels; j++) {
                if (data[j * numBoxes + i] > conf) {
                    conf = data[j * numBoxes + i];
                    cls = j - 4;
                }
            }
        } else {
            float* ptr = data + i * channels;
            bx = ptr[0]; by = ptr[1]; bw = ptr[2]; bh = ptr[3];
            for (int j = 4; j < channels; j++) {
                if (ptr[j] > conf) {
                    conf = ptr[j];
                    cls = j - 4;
                }
            }
        }

        if (conf < confThreshold_) continue;

        int x1 = (int)((bx - bw / 2 - padX_) / scaleX_);
        int y1 = (int)((by - bh / 2 - padY_) / scaleY_);
        int x2 = (int)((bx + bw / 2 - padX_) / scaleX_);
        int y2 = (int)((by + bh / 2 - padY_) / scaleY_);

        x1 = std::max(0, std::min(x1, origW));
        y1 = std::max(0, std::min(y1, origH));
        x2 = std::max(0, std::min(x2, origW));
        y2 = std::max(0, std::min(y2, origH));

        dets.push_back({cls, conf, cv::Rect(x1, y1, x2 - x1, y2 - y1)});
    }

    return nms(dets, iouThreshold_);
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& frame)
{
    if (!loaded_) return {};

    letterbox(frame, letterboxBuf_);
    preprocess(letterboxBuf_);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo_, inputBuf_.data(), inputBuf_.size(),
        inputShape_.data(), inputShape_.size());

    auto outputs = session_.Run(
        Ort::RunOptions{}, inNamesC_.data(), &inputTensor, 1,
        outNamesC_.data(), outNamesC_.size());

    return postprocess(outputs, frame.cols, frame.rows);
}

void YOLODetector::setConfThreshold(float t) { confThreshold_ = t; }
float YOLODetector::confThreshold() const { return confThreshold_; }
void YOLODetector::setIouThreshold(float t) { iouThreshold_ = t; }
float YOLODetector::iouThreshold() const { return iouThreshold_; }
bool YOLODetector::isGpuEnabled() const { return gpuEnabled_; }
bool YOLODetector::isLoaded() const { return loaded_; }
