#include "calibrationdialog.h"
#include "lang.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

CalibrationDialog::CalibrationDialog(StereoSource* source, QWidget* parent)
    : QDialog(parent), source_(source)
{
    setWindowTitle(Lang::s("calibration"));
    setMinimumSize(800, 500);

    stack_ = new QStackedWidget(this);

    // Page 0: Setup
    auto* setupWidget = new QWidget;
    auto* setupLayout = new QVBoxLayout(setupWidget);
    setupLayout->addWidget(new QLabel(Lang::s("calib_board_cols")));
    boardWidthSpin_ = new QSpinBox; boardWidthSpin_->setRange(3, 30); boardWidthSpin_->setValue(9);
    setupLayout->addWidget(boardWidthSpin_);
    setupLayout->addWidget(new QLabel(Lang::s("calib_board_rows")));
    boardHeightSpin_ = new QSpinBox; boardHeightSpin_->setRange(3, 30); boardHeightSpin_->setValue(6);
    setupLayout->addWidget(boardHeightSpin_);
    setupLayout->addWidget(new QLabel(Lang::s("calib_square_size")));
    squareSizeSpin_ = new QDoubleSpinBox; squareSizeSpin_->setRange(1.0, 200.0); squareSizeSpin_->setValue(25.0);
    squareSizeSpin_->setSuffix(" mm");
    setupLayout->addWidget(squareSizeSpin_);
    setupLayout->addStretch();
    auto* startBtn = new QPushButton(Lang::s("calib_start"));
    connect(startBtn, &QPushButton::clicked, this, &CalibrationDialog::onStartCapture);
    setupLayout->addWidget(startBtn);
    stack_->addWidget(setupWidget);

    // Page 1: Capture
    auto* captureWidget = new QWidget;
    auto* captureLayout = new QVBoxLayout(captureWidget);
    auto* previewLayout = new QHBoxLayout;
    leftPreview_ = new QLabel; leftPreview_->setMinimumSize(320, 240);
    rightPreview_ = new QLabel; rightPreview_->setMinimumSize(320, 240);
    leftPreview_->setAlignment(Qt::AlignCenter);
    rightPreview_->setAlignment(Qt::AlignCenter);
    previewLayout->addWidget(leftPreview_);
    previewLayout->addWidget(rightPreview_);
    captureLayout->addLayout(previewLayout);

    captureCountLabel_ = new QLabel(Lang::s("calib_captured").arg(0).arg(MIN_CAPTURE_FRAMES));
    captureLayout->addWidget(captureCountLabel_);

    auto* captureBtnLayout = new QHBoxLayout;
    grabBtn_ = new QPushButton(Lang::s("calib_capture"));
    connect(grabBtn_, &QPushButton::clicked, this, &CalibrationDialog::onGrabFrame);
    captureBtnLayout->addWidget(grabBtn_);

    auto* calibrateBtn = new QPushButton(Lang::s("calib_done"));
    calibrateBtn->setEnabled(false);
    connect(calibrateBtn, &QPushButton::clicked, this, &CalibrationDialog::onCalibrate);
    captureBtnLayout->addWidget(calibrateBtn);
    // Enable calibrate button when enough frames captured
    connect(this, &CalibrationDialog::capturedCountChanged, this, [this, calibrateBtn]() {
        calibrateBtn->setEnabled(capturedCount_ >= MIN_CAPTURE_FRAMES);
    });

    auto* loadExtBtn = new QPushButton(Lang::s("calib_load_external"));
    connect(loadExtBtn, &QPushButton::clicked, this, &CalibrationDialog::onLoadExternal);
    captureBtnLayout->addWidget(loadExtBtn);
    captureLayout->addLayout(captureBtnLayout);

    previewTimer_ = new QTimer(this);
    connect(previewTimer_, &QTimer::timeout, this, &CalibrationDialog::updatePreview);
    stack_->addWidget(captureWidget);

    // Page 2: Results
    auto* resultWidget = new QWidget;
    auto* resultLayout = new QVBoxLayout(resultWidget);
    errorLabel_ = new QLabel;
    errorLabel_->setAlignment(Qt::AlignCenter);
    errorLabel_->setStyleSheet("font-size: 16pt;");
    resultLayout->addWidget(errorLabel_);
    resultLayout->addStretch();

    auto* resultBtnLayout = new QHBoxLayout;
    saveBtn_ = new QPushButton(Lang::s("calib_save"));
    connect(saveBtn_, &QPushButton::clicked, this, &CalibrationDialog::onSaveCalibration);
    resultBtnLayout->addWidget(saveBtn_);

    auto* closeBtn = new QPushButton(tr("Close"));
    connect(closeBtn, &QPushButton::clicked, this, &QDialog::accept);
    resultBtnLayout->addWidget(closeBtn);
    resultLayout->addLayout(resultBtnLayout);
    stack_->addWidget(resultWidget);

    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(stack_);
}

StereoCalibration CalibrationDialog::result() const { return result_; }

void CalibrationDialog::onStartCapture()
{
    capturedCount_ = 0;
    objectPoints_.clear();
    imagePointsL_.clear();
    imagePointsR_.clear();
    captureCountLabel_->setText(Lang::s("calib_captured").arg(0).arg(MIN_CAPTURE_FRAMES));
    stack_->setCurrentIndex(1);
    previewTimer_->start(100);
}

void CalibrationDialog::updatePreview()
{
    if (!source_ || !source_->isOpened()) return;

    cv::Mat left, right;
    if (!source_->grab(left, right)) return;

    // Show scaled previews
    int maxW = leftPreview_->width(), maxH = leftPreview_->height();
    double scale = std::min((double)maxW / left.cols, (double)maxH / left.rows);
    if (scale < 1.0) {
        cv::resize(left, left, cv::Size(), scale, scale);
        cv::resize(right, right, cv::Size(), scale, scale);
    }

    cv::cvtColor(left, left, cv::COLOR_BGR2RGB);
    cv::cvtColor(right, right, cv::COLOR_BGR2RGB);
    QImage imgL(left.data, left.cols, left.rows, left.step, QImage::Format_RGB888);
    QImage imgR(right.data, right.cols, right.rows, right.step, QImage::Format_RGB888);
    leftPreview_->setPixmap(QPixmap::fromImage(imgL.copy()));
    rightPreview_->setPixmap(QPixmap::fromImage(imgR.copy()));
}

void CalibrationDialog::onGrabFrame()
{
    if (!source_ || !source_->isOpened()) return;

    cv::Mat left, right;
    if (!source_->grab(left, right)) return;

    std::vector<cv::Point2f> cornersL, cornersR;
    if (!detectCorners(left, right, cornersL, cornersR)) {
        QMessageBox::warning(this, Lang::s("calibration"),
            Lang::s("calib_no_corners"));
        return;
    }

    imageSize_ = left.size();

    int bw = boardWidthSpin_->value();
    int bh = boardHeightSpin_->value();
    float sqSize = (float)squareSizeSpin_->value();

    std::vector<cv::Point3f> objPts;
    for (int i = 0; i < bh; i++)
        for (int j = 0; j < bw; j++)
            objPts.push_back(cv::Point3f(j * sqSize, i * sqSize, 0));

    objectPoints_.push_back(objPts);
    imagePointsL_.push_back(cornersL);
    imagePointsR_.push_back(cornersR);

    capturedCount_++;
    captureCountLabel_->setText(
        Lang::s("calib_captured").arg(capturedCount_).arg(MIN_CAPTURE_FRAMES));
    emit capturedCountChanged();
}

bool CalibrationDialog::detectCorners(const cv::Mat& left, const cv::Mat& right,
                                       std::vector<cv::Point2f>& cornersL,
                                       std::vector<cv::Point2f>& cornersR)
{
    cv::Size boardSize(boardWidthSpin_->value(), boardHeightSpin_->value());

    int flags = cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE;
    if (!cv::findChessboardCorners(left, boardSize, cornersL, flags))
        return false;
    if (!cv::findChessboardCorners(right, boardSize, cornersR, flags))
        return false;

    cv::Mat grayL, grayR;
    cv::cvtColor(left, grayL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, grayR, cv::COLOR_BGR2GRAY);

    cv::cornerSubPix(grayL, cornersL, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
    cv::cornerSubPix(grayR, cornersR, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

    return true;
}

void CalibrationDialog::onCalibrate()
{
    previewTimer_->stop();
    grabBtn_->setEnabled(false);

    cv::Mat K1, D1, K2, D2, R, T;
    K1 = cv::Mat::eye(3, 3, CV_64F);
    K2 = cv::Mat::eye(3, 3, CV_64F);

    double rms = cv::stereoCalibrate(objectPoints_, imagePointsL_, imagePointsR_,
        K1, D1, K2, D2, imageSize_, R, T,
        cv::noArray(), cv::noArray(),
        cv::CALIB_FIX_INTRINSIC,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6));

    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K1, D1, K2, D2, imageSize_, R, T, R1, R2, P1, P2, Q,
        cv::CALIB_ZERO_DISPARITY, -1, imageSize_);

    result_.cameraMatrixL = K1;
    result_.distCoeffsL = D1;
    result_.cameraMatrixR = K2;
    result_.distCoeffsR = D2;
    result_.R = R;
    result_.T = T;
    result_.R1 = R1;
    result_.R2 = R2;
    result_.P1 = P1;
    result_.P2 = P2;
    result_.Q = Q;
    result_.reprojectionError = rms;
    result_.valid = true;

    stack_->setCurrentIndex(2);
    errorLabel_->setText(Lang::s("calib_error").arg(rms, 0, 'f', 2));

    if (rms < 1.0)
        errorLabel_->setStyleSheet("font-size: 16pt; color: green;");
    else if (rms < 2.0)
        errorLabel_->setStyleSheet("font-size: 16pt; color: orange;");
    else
        errorLabel_->setStyleSheet("font-size: 16pt; color: red;");
}

void CalibrationDialog::onSaveCalibration()
{
    QString path = QFileDialog::getSaveFileName(this, Lang::s("calib_save"),
        "stereo_calib.yml", "YAML (*.yml *.yaml *.xml)",
        nullptr, QFileDialog::DontUseNativeDialog);
    if (path.isEmpty()) return;

    StereoRectifier rectifier;
    rectifier.setCalibration(result_);
    if (rectifier.saveCalibration(path.toStdString()))
        QMessageBox::information(this, Lang::s("calibration"),
            Lang::s("calib_saved") + path);
}

void CalibrationDialog::onLoadExternal()
{
    QString path = QFileDialog::getOpenFileName(this, Lang::s("calib_load_external"),
        "", Lang::s("calib_file_filter"),
        nullptr, QFileDialog::DontUseNativeDialog);
    if (path.isEmpty()) return;

    StereoRectifier rectifier;
    if (rectifier.loadCalibration(path.toStdString())) {
        result_ = rectifier.calibration();
        stack_->setCurrentIndex(2);
        errorLabel_->setText(Lang::s("calib_loaded") + path);
        errorLabel_->setStyleSheet("font-size: 14pt; color: blue;");
    } else {
        QMessageBox::warning(this, Lang::s("error"), Lang::s("calib_load_fail"));
    }
}
