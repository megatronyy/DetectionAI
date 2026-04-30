#ifndef CALIBRATIONDIALOG_H
#define CALIBRATIONDIALOG_H

#include <QDialog>
#include <QStackedWidget>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <vector>
#include <opencv2/opencv.hpp>
#include "stereosource.h"
#include "stereotypes.h"

class CalibrationDialog : public QDialog
{
    Q_OBJECT
public:
    explicit CalibrationDialog(StereoSource* source, QWidget* parent = nullptr);
    StereoCalibration result() const;

signals:
    void capturedCountChanged();

private slots:
    void onStartCapture();
    void onGrabFrame();
    void onCalibrate();
    void onSaveCalibration();
    void onLoadExternal();
    void updatePreview();

private:
    StereoSource* source_;

    QStackedWidget* stack_;
    QSpinBox* boardWidthSpin_;
    QSpinBox* boardHeightSpin_;
    QDoubleSpinBox* squareSizeSpin_;

    QLabel* leftPreview_;
    QLabel* rightPreview_;
    QPushButton* grabBtn_;
    QLabel* captureCountLabel_;
    int capturedCount_ = 0;
    static const int MIN_CAPTURE_FRAMES = 15;

    QLabel* errorLabel_;
    QPushButton* saveBtn_;
    QPushButton* loadExternalBtn_;

    std::vector<std::vector<cv::Point3f>> objectPoints_;
    std::vector<std::vector<cv::Point2f>> imagePointsL_, imagePointsR_;
    cv::Size imageSize_;
    StereoCalibration result_;

    QTimer* previewTimer_;

    bool detectCorners(const cv::Mat& left, const cv::Mat& right,
                       std::vector<cv::Point2f>& cornersL,
                       std::vector<cv::Point2f>& cornersR);
};

#endif // CALIBRATIONDIALOG_H
