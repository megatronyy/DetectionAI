#include "inferencethread.h"
#include <QElapsedTimer>

InferenceThread::InferenceThread(QObject* parent) : QThread(parent) {}

InferenceThread::~InferenceThread()
{
    stop();
    wait();
}

YOLODetector& InferenceThread::detector() { return detector_; }
bool InferenceThread::isVideoSource() const { return isVideo_; }
void InferenceThread::setPaused(bool p) { paused_ = p; }
bool InferenceThread::isRecording() const { return recording_; }
void InferenceThread::setTrackingEnabled(bool e) { trackingEnabled_ = e; }
bool InferenceThread::isTrackingEnabled() const { return trackingEnabled_; }
void InferenceThread::resetTracker() { tracker_.reset(); }
void InferenceThread::setLoopEnabled(bool e) { loopEnabled_ = e; }
bool InferenceThread::isLoopEnabled() const { return loopEnabled_; }
QSize InferenceThread::frameSize() const { return lastFrameSize_; }
std::vector<Detection> InferenceThread::lastDetections() const {
    std::lock_guard<std::mutex> lock(detectionsMutex_);
    return lastDetections_;
}

bool InferenceThread::openCamera(int index)
{
    if (cap_.isOpened()) cap_.release();
    cap_.open(index);
    isVideo_ = false;
    tracker_.reset();
    return cap_.isOpened();
}

bool InferenceThread::openVideo(const std::string& path)
{
    if (cap_.isOpened()) cap_.release();
    cap_.open(path);
    isVideo_ = true;
    tracker_.reset();

    if (path.compare(0, 7, "rtsp://") == 0) {
        cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }

    return cap_.isOpened();
}

void InferenceThread::stop()
{
    running_ = false;
}

void InferenceThread::startRecording(const std::string& path, double fps, int width, int height)
{
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    writer_.open(path, fourcc, fps, cv::Size(width, height));
    recording_ = writer_.isOpened();
}

void InferenceThread::stopRecording()
{
    recording_ = false;
    if (writer_.isOpened()) writer_.release();
}

cv::Scalar InferenceThread::classColor(int classId)
{
    // Golden angle for perceptually distinct colors
    float hue = fmod(classId * 137.508f, 180.f); // OpenCV Hue range: 0-180
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((int)hue, 200, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    auto c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
}

void InferenceThread::drawDetections(cv::Mat& frame, const std::vector<Detection>& dets)
{
    for (const auto& d : dets) {
        cv::Scalar color = classColor(d.classId);
        cv::rectangle(frame, d.bbox, color, 2);

        std::string label = YOLODetector::CLASS_NAMES[d.classId] +
                            " " + cv::format("%.2f", d.confidence);
        int baseline = 0;
        double fontScale = 0.5;
        int thickness = 1;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             fontScale, thickness, &baseline);

        int top = d.bbox.y;
        int textY = (top > textSize.height + 4) ? (top - 4) : (top + textSize.height + 4);

        cv::rectangle(frame,
            cv::Point(d.bbox.x, textY - textSize.height - 2),
            cv::Point(d.bbox.x + textSize.width, textY + 2),
            color, cv::FILLED);
        cv::putText(frame, label,
            cv::Point(d.bbox.x, textY),
            cv::FONT_HERSHEY_SIMPLEX, fontScale,
            cv::Scalar(0, 0, 0), thickness);
    }
}

void InferenceThread::drawTracks(cv::Mat& frame, const std::vector<Track>& tracks)
{
    for (const auto& t : tracks) {
        cv::Scalar color = classColor(t.det.classId);
        cv::rectangle(frame, t.det.bbox, color, 2);

        std::string label = "#" + std::to_string(t.trackId) + " " +
                            YOLODetector::CLASS_NAMES[t.det.classId] +
                            " " + cv::format("%.2f", t.det.confidence);
        int baseline = 0;
        double fontScale = 0.5;
        int thickness = 1;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             fontScale, thickness, &baseline);

        int top = t.det.bbox.y;
        int textY = (top > textSize.height + 4) ? (top - 4) : (top + textSize.height + 4);

        cv::rectangle(frame,
            cv::Point(t.det.bbox.x, textY - textSize.height - 2),
            cv::Point(t.det.bbox.x + textSize.width, textY + 2),
            color, cv::FILLED);
        cv::putText(frame, label,
            cv::Point(t.det.bbox.x, textY),
            cv::FONT_HERSHEY_SIMPLEX, fontScale,
            cv::Scalar(0, 0, 0), thickness);
    }
}

void InferenceThread::run()
{
    running_ = true;
    QElapsedTimer fpsTimer;
    fpsTimer.start();
    int frameCount = 0;
    float fps = 0;
    int emptyCount = 0;

    while (running_) {
        if (paused_) {
            msleep(30);
            continue;
        }

        if (!cap_.isOpened()) {
            msleep(100);
            continue;
        }

        cap_ >> currentFrame_;
        if (currentFrame_.empty()) {
            emptyCount++;
            if (isVideo_ && emptyCount > 5) {
                if (loopEnabled_) {
                    cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
                    emptyCount = 0;
                    continue;
                }
                emit inputLost("视频播放结束");
                paused_ = true;
                continue;
            }
            if (!isVideo_ && emptyCount > 30) {
                emit inputLost("摄像头已断开");
                paused_ = true;
                continue;
            }
            msleep(10);
            continue;
        }
        emptyCount = 0;
        lastFrameSize_ = QSize(currentFrame_.cols, currentFrame_.rows);

        QElapsedTimer inferTimer;
        inferTimer.start();
        auto dets = detector_.detect(currentFrame_);
        float inferMs = (float)inferTimer.elapsed();

        {
            std::lock_guard<std::mutex> lock(detectionsMutex_);
            lastDetections_ = dets;
        }

        if (trackingEnabled_) {
            auto tracks = tracker_.update(dets);
            drawTracks(currentFrame_, tracks);
        } else {
            drawDetections(currentFrame_, dets);
        }

        if (recording_ && writer_.isOpened()) {
            writer_ << currentFrame_;
        }

        // Deep copy QImage for thread safety
        QImage img;
        if (currentFrame_.channels() == 3)
            img = QImage(currentFrame_.data, currentFrame_.cols, currentFrame_.rows,
                         currentFrame_.step, QImage::Format_BGR888).copy();

        frameCount++;
        if (fpsTimer.elapsed() >= 1000) {
            fps = frameCount * 1000.f / fpsTimer.elapsed();
            frameCount = 0;
            fpsTimer.restart();
        }

        QMap<int,int> classCounts;
        for (const auto& d : dets) classCounts[d.classId]++;

        emit frameReady(img, (int)dets.size(), fps, inferMs, classCounts);
    }

    if (writer_.isOpened()) writer_.release();
}
