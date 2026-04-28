#include "inferencethread.h"
#include <QElapsedTimer>
#include <QDateTime>

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
void InferenceThread::resetTrackCounts() { tracker_.resetCounts(); }
void InferenceThread::setTrajectoryEnabled(bool e) { trajectoryEnabled_ = e; }
bool InferenceThread::isTrajectoryEnabled() const { return trajectoryEnabled_; }
void InferenceThread::setSpeedEnabled(bool e) { speedEnabled_ = e; }
bool InferenceThread::isSpeedEnabled() const { return speedEnabled_; }
void InferenceThread::setCountingLine(const CountingLine& line) { tracker_.setCountingLine(line); }
void InferenceThread::clearCountingLine() { tracker_.clearCountingLine(); }
bool InferenceThread::hasCountingLine() const { return tracker_.hasCountingLine(); }
QMap<int, QMap<int, int>> InferenceThread::crossingCountsByDir() const { return tracker_.crossingCountsByDir(); }
void InferenceThread::resetCrossingCounts() { tracker_.resetCrossingCounts(); }
void InferenceThread::setLoopEnabled(bool e) { loopEnabled_ = e; }
bool InferenceThread::isLoopEnabled() const { return loopEnabled_; }
QSize InferenceThread::frameSize() const { return lastFrameSize_; }
std::vector<Detection> InferenceThread::lastDetections() const {
    std::lock_guard<std::mutex> lock(detectionsMutex_);
    return lastDetections_;
}
std::vector<TrackRecord> InferenceThread::trackHistory() const {
    std::lock_guard<std::mutex> lock(historyMutex_);
    return trackHistory_;
}
void InferenceThread::clearTrackHistory() {
    std::lock_guard<std::mutex> lock(historyMutex_);
    trackHistory_.clear();
}

bool InferenceThread::openCamera(int index)
{
    if (cap_.isOpened()) cap_.release();
    cap_.open(index);
    isVideo_ = false;
    tracker_.reset();
    clearTrackHistory();
    return cap_.isOpened();
}

bool InferenceThread::openVideo(const std::string& path)
{
    if (cap_.isOpened()) cap_.release();
    cap_.open(path);
    isVideo_ = true;
    tracker_.reset();
    clearTrackHistory();

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

void InferenceThread::drawLabel(cv::Mat& frame, const cv::Rect& bbox,
                                 int classId, const std::string& label)
{
    if (classId < 0 || classId >= YOLODetector::NUM_CLASSES) return;
    cv::Scalar color = classColor(classId);
    cv::rectangle(frame, bbox, color, 2);

    int baseline = 0;
    double fontScale = 0.5;
    int thickness = 1;
    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                         fontScale, thickness, &baseline);
    int top = bbox.y;
    int textY = (top > textSize.height + 4) ? (top - 4) : (top + textSize.height + 4);

    cv::rectangle(frame,
        cv::Point(bbox.x, textY - textSize.height - 2),
        cv::Point(bbox.x + textSize.width, textY + 2),
        color, cv::FILLED);
    cv::putText(frame, label,
        cv::Point(bbox.x, textY),
        cv::FONT_HERSHEY_SIMPLEX, fontScale,
        cv::Scalar(0, 0, 0), thickness);
}

void InferenceThread::drawDetections(cv::Mat& frame, const std::vector<Detection>& dets)
{
    for (const auto& d : dets) {
        std::string label = YOLODetector::CLASS_NAMES[d.classId] +
                            " " + cv::format("%.2f", d.confidence);
        drawLabel(frame, d.bbox, d.classId, label);
    }
}

void InferenceThread::drawTracks(cv::Mat& frame, const std::vector<Track>& tracks)
{
    for (const auto& t : tracks) {
        std::string label = "#" + std::to_string(t.trackId) + " " +
                            YOLODetector::CLASS_NAMES[t.det.classId] +
                            " " + cv::format("%.2f", t.det.confidence);
        if (speedEnabled_ && t.speed > 0.5f)
            label += " " + cv::format("%.1fpx/f", t.speed);
        drawLabel(frame, t.det.bbox, t.det.classId, label);

        if (speedEnabled_ && t.speed > 0.5f) {
            cv::Point center(t.det.bbox.x + t.det.bbox.width / 2,
                             t.det.bbox.y + t.det.bbox.height / 2);
            float rad = t.angle * (float)CV_PI / 180.f;
            int arrowLen = std::min(30, (int)(t.speed * 3 + 10));
            cv::Point endPt(center.x + (int)(arrowLen * std::cos(rad)),
                            center.y + (int)(arrowLen * std::sin(rad)));
            cv::Scalar color = classColor(t.det.classId);
            cv::arrowedLine(frame, center, endPt, color, 2, cv::LINE_8, 0, 0.3);
        }
    }
}

void InferenceThread::drawTrajectory(cv::Mat& frame, const std::vector<Track>& tracks)
{
    for (const auto& t : tracks) {
        if (t.trajectory.size() < 2) continue;
        cv::Scalar color = classColor(t.det.classId);
        for (size_t i = 1; i < t.trajectory.size(); i++) {
            int thickness = std::max(1, (int)(i * 2 / t.trajectory.size()));
            cv::line(frame, t.trajectory[i - 1], t.trajectory[i], color, thickness);
        }
    }
}

void InferenceThread::drawCountingLine(cv::Mat& frame)
{
    if (!tracker_.hasCountingLine()) return;
    auto line = tracker_.countingLine();
    cv::Scalar lineColor(0, 255, 255);

    // Dashed line
    cv::Point dir = line.pt2 - line.pt1;
    double length = std::sqrt((double)dir.x * dir.x + dir.y * dir.y);
    if (length < 1.0) return;
    double dx = dir.x / length, dy = dir.y / length;
    int dashLen = 10;
    int numDashes = (int)(length / dashLen);
    for (int i = 0; i < numDashes; i += 2) {
        double s = i * dashLen, e = std::min((i + 1.0) * dashLen, length);
        cv::line(frame,
            cv::Point(line.pt1.x + (int)(dx * s), line.pt1.y + (int)(dy * s)),
            cv::Point(line.pt1.x + (int)(dx * e), line.pt1.y + (int)(dy * e)),
            lineColor, 2);
    }

    // End points
    cv::circle(frame, line.pt1, 5, lineColor, -1);
    cv::circle(frame, line.pt2, 5, lineColor, -1);

    // Label
    if (!line.label.empty()) {
        cv::Point mid((line.pt1.x + line.pt2.x) / 2, (line.pt1.y + line.pt2.y) / 2);
        cv::putText(frame, line.label, cv::Point(mid.x, mid.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, lineColor, 2);
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
            tracker_.setCurrentTime(QDateTime::currentMSecsSinceEpoch());
            auto tracks = tracker_.update(dets);
            drawTracks(currentFrame_, tracks);
            if (trajectoryEnabled_)
                drawTrajectory(currentFrame_, tracks);
            if (tracker_.hasCountingLine())
                drawCountingLine(currentFrame_);

            // Record track history
            {
                int64_t nowMs = QDateTime::currentMSecsSinceEpoch();
                std::lock_guard<std::mutex> lock(historyMutex_);
                for (const auto& t : tracks) {
                    TrackRecord rec;
                    rec.trackId = t.trackId;
                    rec.classId = t.det.classId;
                    rec.timestampMs = nowMs;
                    rec.x = t.det.bbox.x;
                    rec.y = t.det.bbox.y;
                    rec.width = t.det.bbox.width;
                    rec.height = t.det.bbox.height;
                    rec.speed = t.speed;
                    rec.angle = t.angle;
                    trackHistory_.push_back(rec);
                }
                if ((int)trackHistory_.size() > MAX_HISTORY)
                    trackHistory_.erase(trackHistory_.begin(),
                        trackHistory_.begin() + (trackHistory_.size() - MAX_HISTORY));
            }
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
            if (trackingEnabled_) {
                auto uc = tracker_.uniqueCounts();
                emit trackingStatsUpdated(uc, tracker_.totalUnique());
                if (tracker_.hasCountingLine())
                    emit crossingStatsUpdated(tracker_.crossingCountsByDir());
            }
        }

        QMap<int,int> classCounts;
        for (const auto& d : dets) classCounts[d.classId]++;

        emit frameReady(img, (int)dets.size(), fps, inferMs, classCounts);
    }

    if (writer_.isOpened()) writer_.release();
}
