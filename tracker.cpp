#include "tracker.h"
#include <algorithm>
#include <cmath>

Tracker::Tracker(int maxAge, int minHits, float iouThreshold)
    : maxAge_(maxAge), minHits_(minHits), iouThreshold_(iouThreshold)
{
}

void Tracker::reset()
{
    tracks_.clear();
    nextId_ = 0;
    seenIdsByClass_.clear();
}

void Tracker::resetCounts()
{
    seenIdsByClass_.clear();
}

QMap<int, int> Tracker::uniqueCounts() const
{
    QMap<int, int> counts;
    for (auto it = seenIdsByClass_.constBegin(); it != seenIdsByClass_.constEnd(); ++it)
        counts[it.key()] = it.value().size();
    return counts;
}

int Tracker::totalUnique() const
{
    int total = 0;
    for (auto it = seenIdsByClass_.constBegin(); it != seenIdsByClass_.constEnd(); ++it)
        total += it.value().size();
    return total;
}

void Tracker::setCountingLine(const CountingLine& line)
{
    countLine_ = line;
    hasLine_ = true;
    for (auto& t : tracks_) {
        cv::Point center(t.lastBbox.x + t.lastBbox.width / 2,
                         t.lastBbox.y + t.lastBbox.height / 2);
        t.lastSide = sideOfLine(center, countLine_);
    }
}

void Tracker::clearCountingLine()
{
    hasLine_ = false;
    countLine_ = CountingLine();
    for (auto& t : tracks_) t.lastSide = 0;
}

bool Tracker::hasCountingLine() const { return hasLine_; }
CountingLine Tracker::countingLine() const { return countLine_; }
QMap<int, QMap<int, int>> Tracker::crossingCountsByDir() const { return crossingCountsByDir_; }

void Tracker::resetCrossingCounts()
{
    crossings_.clear();
    crossingCountsByDir_.clear();
}

void Tracker::setCurrentTime(int64_t ms) { currentTimeMs_ = ms; }

float Tracker::computeIoU(const cv::Rect& a, const cv::Rect& b)
{
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    int inter = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.width * a.height + b.width * b.height - inter;
    return unionArea > 0 ? (float)inter / unionArea : 0.f;
}

int Tracker::sideOfLine(const cv::Point& p, const CountingLine& line)
{
    long long cross = (long long)(line.pt2.x - line.pt1.x) * (p.y - line.pt1.y)
                    - (long long)(line.pt2.y - line.pt1.y) * (p.x - line.pt1.x);
    if (cross > 0) return 1;
    if (cross < 0) return -1;
    return 0;
}

cv::Mat Tracker::bboxToMeasurement(const cv::Rect& bbox)
{
    float cx = bbox.x + bbox.width / 2.f;
    float cy = bbox.y + bbox.height / 2.f;
    float s = (float)bbox.width * bbox.height;
    float r = bbox.height > 0 ? (float)bbox.width / bbox.height : 1.f;
    cv::Mat m = (cv::Mat_<float>(4, 1) << cx, cy, s, r);
    return m;
}

cv::Rect Tracker::stateToBbox(const cv::Mat& state)
{
    float cx = state.at<float>(0);
    float cy = state.at<float>(1);
    float s  = state.at<float>(2);
    float r  = state.at<float>(3);

    s = std::max(s, 1.f);
    r = std::max(r, 0.1f);

    float w = std::sqrt(s * r);
    float h = s / w;
    return cv::Rect((int)(cx - w / 2), (int)(cy - h / 2), (int)w, (int)h);
}

cv::KalmanFilter Tracker::createKalmanFilter(const cv::Rect& bbox)
{
    // State: [cx, cy, s, r, vcx, vcy, vs], Measurement: [cx, cy, s, r]
    cv::KalmanFilter kf(7, 4, 0);

    // Transition matrix
    cv::setIdentity(kf.transitionMatrix);
    kf.transitionMatrix.at<float>(4, 0) = 1.f;  // vcx -> cx
    kf.transitionMatrix.at<float>(5, 1) = 1.f;  // vcy -> cy
    kf.transitionMatrix.at<float>(6, 2) = 1.f;  // vs  -> s

    // Measurement matrix
    kf.measurementMatrix = cv::Mat::zeros(4, 7, CV_32F);
    kf.measurementMatrix.at<float>(0, 0) = 1.f;
    kf.measurementMatrix.at<float>(1, 1) = 1.f;
    kf.measurementMatrix.at<float>(2, 2) = 1.f;
    kf.measurementMatrix.at<float>(3, 3) = 1.f;

    // Process noise
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    kf.processNoiseCov.at<float>(4, 4) = 1e-3f;
    kf.processNoiseCov.at<float>(5, 5) = 1e-3f;
    kf.processNoiseCov.at<float>(6, 6) = 1e-3f;

    // Measurement noise
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));

    // Posterior error covariance
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(10));

    // Initialize state
    cv::Mat m = bboxToMeasurement(bbox);
    kf.statePost.at<float>(0) = m.at<float>(0);
    kf.statePost.at<float>(1) = m.at<float>(1);
    kf.statePost.at<float>(2) = m.at<float>(2);
    kf.statePost.at<float>(3) = m.at<float>(3);
    kf.statePost.at<float>(4) = 0.f;
    kf.statePost.at<float>(5) = 0.f;
    kf.statePost.at<float>(6) = 0.f;

    return kf;
}

std::vector<int> Tracker::hungarian(const cv::Mat& costMatrix, float iouThresh)
{
    int rows = costMatrix.rows;
    int cols = costMatrix.cols;
    int n = std::max(rows, cols);

    if (rows == 0 || cols == 0) return std::vector<int>(rows, -1);

    // Work on a square cost matrix padded with large values
    cv::Mat cost = cv::Mat::ones(n, n, CV_32F) * 1e6f;
    costMatrix.copyTo(cost(cv::Rect(0, 0, cols, rows)));

    std::vector<float> u(n + 1), v(n + 1);
    std::vector<int> p(n + 1), way(n + 1);

    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(n + 1, 1e9f);
        std::vector<bool> used(n + 1, false);

        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            float delta = 1e9f;

            for (int j = 1; j <= n; j++) {
                if (!used[j]) {
                    float cur = cost.at<float>(i0 - 1, j - 1) - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for (int j = 0; j <= n; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    // p[j] = row assigned to column j (1-indexed)
    std::vector<int> assignment(rows, -1);
    for (int j = 1; j <= cols; j++) {
        if (p[j] >= 1 && p[j] <= rows) {
            // Only assign if cost is below threshold (not a padded dummy)
            if (cost.at<float>(p[j] - 1, j - 1) < iouThresh) {
                assignment[p[j] - 1] = j - 1;
            }
        }
    }

    return assignment;
}

std::vector<Track> Tracker::update(const std::vector<Detection>& detections)
{
    int numTracks = (int)tracks_.size();
    int numDets = (int)detections.size();

    // 1. Predict existing tracks
    std::vector<cv::Rect> predictedBoxes(numTracks);
    for (int i = 0; i < numTracks; i++) {
        cv::Mat pred = tracks_[i].kf.predict();
        predictedBoxes[i] = stateToBbox(pred);
    }

    // 2. Build IoU cost matrix (1 - IoU)
    cv::Mat costMatrix;
    if (numTracks > 0 && numDets > 0) {
        costMatrix = cv::Mat::ones(numTracks, numDets, CV_32F);
        for (int i = 0; i < numTracks; i++) {
            for (int j = 0; j < numDets; j++) {
                costMatrix.at<float>(i, j) = 1.f - computeIoU(predictedBoxes[i], detections[j].bbox);
            }
        }
    }

    // 3. Hungarian assignment
    std::vector<int> assignment;
    if (numTracks > 0 && numDets > 0) {
        assignment = hungarian(costMatrix, iouThreshold_);
    } else {
        assignment.resize(numTracks, -1);
    }

    std::vector<bool> detUsed(numDets, false);

    // 4. Update matched tracks
    for (int i = 0; i < numTracks; i++) {
        int j = assignment[i];
        if (j >= 0 && j < numDets) {
            detUsed[j] = true;
            cv::Mat m = bboxToMeasurement(detections[j].bbox);
            tracks_[i].kf.correct(m);
            tracks_[i].lastBbox = detections[j].bbox;
            tracks_[i].classId = detections[j].classId;
            tracks_[i].confidence = detections[j].confidence;
            tracks_[i].hits++;
            tracks_[i].missedFrames = 0;
            cv::Point center(detections[j].bbox.x + detections[j].bbox.width / 2,
                             detections[j].bbox.y + detections[j].bbox.height / 2);
            tracks_[i].trajectory.push_back(center);
            if ((int)tracks_[i].trajectory.size() > MAX_TRAJECTORY_LEN)
                tracks_[i].trajectory.erase(tracks_[i].trajectory.begin());

            if (hasLine_) {
                int currentSide = sideOfLine(center, countLine_);
                if (tracks_[i].lastSide != 0 && currentSide != 0 && currentSide != tracks_[i].lastSide) {
                    CrossingEvent evt;
                    evt.trackId = tracks_[i].trackId;
                    evt.classId = tracks_[i].classId;
                    evt.direction = currentSide;
                    evt.timestampMs = currentTimeMs_;
                    crossings_.push_back(evt);
                    crossingCountsByDir_[tracks_[i].classId][currentSide]++;
                }
                tracks_[i].lastSide = currentSide;
            }
        } else {
            tracks_[i].missedFrames++;
        }
    }

    // 5. Remove old tracks
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
            [this](const InternalTrack& t) { return t.missedFrames > maxAge_; }),
        tracks_.end());

    // 6. Create new tracks for unmatched detections
    for (int j = 0; j < numDets; j++) {
        if (!detUsed[j]) {
            InternalTrack t;
            t.kf = createKalmanFilter(detections[j].bbox);
            t.trackId = nextId_++;
            t.hits = 1;
            t.missedFrames = 0;
            t.lastBbox = detections[j].bbox;
            t.classId = detections[j].classId;
            t.confidence = detections[j].confidence;
            cv::Point center(detections[j].bbox.x + detections[j].bbox.width / 2,
                             detections[j].bbox.y + detections[j].bbox.height / 2);
            t.trajectory.push_back(center);
            tracks_.push_back(t);
            seenIdsByClass_[detections[j].classId].insert(t.trackId);

            if (hasLine_) {
                cv::Point tc(detections[j].bbox.x + detections[j].bbox.width / 2,
                             detections[j].bbox.y + detections[j].bbox.height / 2);
                tracks_.back().lastSide = sideOfLine(tc, countLine_);
            }
        }
    }

    // 7. Build output
    std::vector<Track> result;
    result.reserve(tracks_.size());
    for (const auto& t : tracks_) {
        if (t.hits >= minHits_ && t.missedFrames == 0) {
            float vx = t.kf.statePost.at<float>(4);
            float vy = t.kf.statePost.at<float>(5);
            float speed = std::sqrt(vx * vx + vy * vy);
            float angle = std::atan2(vy, vx) * 180.f / (float)CV_PI;
            if (angle < 0) angle += 360.f;
            result.push_back({{t.classId, t.confidence, t.lastBbox}, t.trackId, t.trajectory, speed, angle});
        }
    }
    return result;
}
