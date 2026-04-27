#include "mainwindow.h"
#include "classfilterdialog.h"
#include <QIcon>
#include <QToolBar>
#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QHBoxLayout>
#include <QDateTime>
#include <QSettings>
#include <QDebug>
#include <QInputDialog>
#include <QFileInfo>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowIcon(QIcon("app.ico"));
    setWindowTitle("YOLO11 物体检测");
    resize(960, 720);

    setupUI();
    loadSettings();

    // Load model
    QSettings initSettings("DetectionAI", "YOLODetector");
    QString modelPath = initSettings.value("modelPath", "yolo11n.onnx").toString();
    if (!thread_.detector().loadModel(modelPath.toStdWString())) {
        modelPath = QFileDialog::getOpenFileName(this,
            "选择 ONNX 模型", "", "ONNX 模型 (*.onnx)");
        if (modelPath.isEmpty() || !thread_.detector().loadModel(modelPath.toStdWString())) {
            QMessageBox::critical(this, "错误", "无法加载 ONNX 模型。");
            return;
        }
    }
    currentModelPath_ = modelPath;

    deviceLabel_->setText(thread_.detector().isGpuEnabled() ? "GPU (CUDA)" : "CPU");

    // Open default camera
    int cam = cameraCombo_->currentData().toInt();
    if (!thread_.openCamera(cam)) {
        QMessageBox::warning(this, "警告",
            "无法打开默认摄像头，请选择其他输入源或打开视频文件。");
    }

    // Connect signals
    connect(&thread_, &InferenceThread::frameReady,
            this, &MainWindow::onFrameReady);
    connect(&thread_, &InferenceThread::inputLost,
            this, &MainWindow::onInputLost);

    thread_.start();
}

MainWindow::~MainWindow()
{
    thread_.stop();
    thread_.wait();
}

void MainWindow::setupUI()
{
    // --- Toolbar ---
    QToolBar* toolbar = addToolBar("Controls");
    toolbar->setMovable(false);

    pauseBtn_ = new QPushButton("暂停");
    screenshotBtn_ = new QPushButton("截屏");
    recordBtn_ = new QPushButton("录制");
    videoBtn_ = new QPushButton("打开视频");
    networkCamBtn_ = new QPushButton("网络摄像头");
    switchModelBtn_ = new QPushButton("切换模型");
    classFilterBtn_ = new QPushButton("类别筛选");
    trackingBtn_ = new QPushButton("目标追踪");
    trackingBtn_->setCheckable(true);

    cameraCombo_ = new QComboBox;
    for (int i = 0; i < 5; i++)
        cameraCombo_->addItem(QString("摄像头 %1").arg(i), i);
    cameraCombo_->setCurrentIndex(0);

    toolbar->addWidget(pauseBtn_);
    toolbar->addSeparator();
    toolbar->addWidget(screenshotBtn_);
    toolbar->addWidget(recordBtn_);
    toolbar->addSeparator();
    toolbar->addWidget(videoBtn_);
    toolbar->addWidget(networkCamBtn_);
    toolbar->addSeparator();
    toolbar->addWidget(switchModelBtn_);
    toolbar->addSeparator();
    toolbar->addWidget(classFilterBtn_);
    toolbar->addWidget(trackingBtn_);
    toolbar->addSeparator();
    toolbar->addWidget(new QLabel(" 输入: "));
    toolbar->addWidget(cameraCombo_);

    connect(pauseBtn_, &QPushButton::clicked, this, &MainWindow::onTogglePause);
    connect(screenshotBtn_, &QPushButton::clicked, this, &MainWindow::onScreenshot);
    connect(recordBtn_, &QPushButton::clicked, this, &MainWindow::onToggleRecord);
    connect(videoBtn_, &QPushButton::clicked, this, &MainWindow::onOpenVideo);
    connect(networkCamBtn_, &QPushButton::clicked, this, &MainWindow::onNetworkCamera);
    connect(switchModelBtn_, &QPushButton::clicked, this, &MainWindow::onSwitchModel);
    connect(classFilterBtn_, &QPushButton::clicked, this, &MainWindow::onClassFilter);
    connect(trackingBtn_, &QPushButton::toggled, this, &MainWindow::onToggleTracking);
    connect(cameraCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onCameraChanged);

    // --- Sliders ---
    QWidget* sliderWidget = new QWidget;
    QHBoxLayout* sliderLayout = new QHBoxLayout(sliderWidget);
    sliderLayout->setContentsMargins(8, 4, 8, 4);

    // Confidence slider
    sliderLayout->addWidget(new QLabel("置信度:"));
    confSlider_ = new QSlider(Qt::Horizontal);
    confSlider_->setRange(1, 99);
    confSlider_->setValue(25);
    confSlider_->setTickPosition(QSlider::TicksBelow);
    confSlider_->setTickInterval(10);
    confSlider_->setMinimumWidth(150);
    confValueLabel_ = new QLabel("0.25");
    confValueLabel_->setMinimumWidth(36);
    sliderLayout->addWidget(confSlider_);
    sliderLayout->addWidget(confValueLabel_);

    sliderLayout->addSpacing(20);

    // IoU slider
    sliderLayout->addWidget(new QLabel("IoU:"));
    iouSlider_ = new QSlider(Qt::Horizontal);
    iouSlider_->setRange(10, 90);
    iouSlider_->setValue(45);
    iouSlider_->setTickPosition(QSlider::TicksBelow);
    iouSlider_->setTickInterval(10);
    iouSlider_->setMinimumWidth(150);
    iouValueLabel_ = new QLabel("0.45");
    iouValueLabel_->setMinimumWidth(36);
    sliderLayout->addWidget(iouSlider_);
    sliderLayout->addWidget(iouValueLabel_);

    sliderLayout->addStretch();

    connect(confSlider_, &QSlider::valueChanged, this, &MainWindow::onConfChanged);
    connect(iouSlider_, &QSlider::valueChanged, this, &MainWindow::onIouChanged);

    // --- Video display ---
    videoLabel_ = new QLabel;
    videoLabel_->setAlignment(Qt::AlignCenter);
    videoLabel_->setStyleSheet("QLabel { background-color: #1a1a1a; }");

    // --- Layout ---
    QWidget* central = new QWidget;
    QVBoxLayout* mainLayout = new QVBoxLayout(central);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);
    mainLayout->addWidget(sliderWidget);
    mainLayout->addWidget(videoLabel_, 1);
    setCentralWidget(central);

    // --- Status bar ---
    fpsLabel_ = new QLabel("FPS: --");
    detLabel_ = new QLabel("检测数: 0");
    deviceLabel_ = new QLabel("CPU");

    statusBar()->addWidget(fpsLabel_);
    statusBar()->addWidget(detLabel_);
    statusBar()->addPermanentWidget(deviceLabel_);
}

void MainWindow::loadSettings()
{
    QSettings settings("DetectionAI", "YOLODetector");
    float conf = settings.value("confidence", 0.25).toFloat();
    float iou = settings.value("iou", 0.45).toFloat();
    int cam = settings.value("camera", 0).toInt();
    QSize winSize = settings.value("windowSize", QSize(960, 720)).toSize();
    bool tracking = settings.value("tracking", false).toBool();

    confSlider_->setValue((int)(conf * 100));
    iouSlider_->setValue((int)(iou * 100));
    cameraCombo_->setCurrentIndex(std::min(cam, cameraCombo_->count() - 1));
    resize(winSize);

    trackingBtn_->setChecked(tracking);
    thread_.setTrackingEnabled(tracking);

    QList<QVariant> classList = settings.value("enabledClasses").toList();
    QSet<int> classes;
    for (const auto& v : classList) classes.insert(v.toInt());
    thread_.detector().setEnabledClasses(classes);
    enabledClasses_ = classes;

    thread_.detector().setConfThreshold(conf);
    thread_.detector().setIouThreshold(iou);
}

void MainWindow::saveSettings()
{
    QSettings settings("DetectionAI", "YOLODetector");
    settings.setValue("confidence", thread_.detector().confThreshold());
    settings.setValue("iou", thread_.detector().iouThreshold());
    settings.setValue("camera", cameraCombo_->currentIndex());
    settings.setValue("windowSize", size());
    settings.setValue("modelPath", currentModelPath_);
    settings.setValue("tracking", thread_.isTrackingEnabled());

    QList<QVariant> classList;
    for (int id : enabledClasses_) classList.append(id);
    settings.setValue("enabledClasses", classList);
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    saveSettings();
    thread_.stop();
    thread_.wait();
    event->accept();
}

void MainWindow::keyPressEvent(QKeyEvent* event)
{
    switch (event->key()) {
    case Qt::Key_Space:
        onTogglePause();
        break;
    case Qt::Key_S:
        onScreenshot();
        break;
    case Qt::Key_O:
        onOpenVideo();
        break;
    case Qt::Key_N:
        onNetworkCamera();
        break;
    case Qt::Key_M:
        onSwitchModel();
        break;
    case Qt::Key_T:
        trackingBtn_->toggle();
        break;
    case Qt::Key_F11:
        isFullScreen() ? showNormal() : showFullScreen();
        break;
    case Qt::Key_Escape:
        if (isFullScreen()) showNormal();
        else close();
        break;
    default:
        QMainWindow::keyPressEvent(event);
    }
}

// --- Slots ---

void MainWindow::onFrameReady(const QImage& image, int detCount, float fps)
{
    if (image.isNull()) return;

    lastFrame_ = image;
    videoLabel_->setPixmap(QPixmap::fromImage(image).scaled(
        videoLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

    fpsLabel_->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
    detLabel_->setText(QString("检测数: %1").arg(detCount));
}

void MainWindow::onInputLost(const QString& msg)
{
    statusBar()->showMessage(msg, 5000);
}

void MainWindow::onTogglePause()
{
    paused_ = !paused_;
    thread_.setPaused(paused_);
    pauseBtn_->setText(paused_ ? "继续" : "暂停");
}

void MainWindow::onScreenshot()
{
    if (lastFrame_.isNull()) return;

    QString defaultName = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".png";
    QString path = QFileDialog::getSaveFileName(this, "保存截图", defaultName,
        "图片 (*.png *.jpg *.bmp)");
    if (path.isEmpty()) return;

    if (lastFrame_.save(path))
        statusBar()->showMessage("截图已保存: " + path, 3000);
    else
        QMessageBox::warning(this, "错误", "保存截图失败。");
}

void MainWindow::onOpenVideo()
{
    QString path = QFileDialog::getOpenFileName(this, "打开视频", "",
        "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv);;所有文件 (*)");
    if (path.isEmpty()) return;

    thread_.stop();
    thread_.wait();

    if (!thread_.openVideo(path.toStdString())) {
        QMessageBox::warning(this, "错误", "无法打开视频文件。");
        return;
    }

    paused_ = false;
    pauseBtn_->setText("暂停");
    thread_.start();
    statusBar()->showMessage("视频: " + path, 3000);
}

void MainWindow::onCameraChanged(int index)
{
    if (index < 0) return;
    int camIdx = cameraCombo_->itemData(index).toInt();

    thread_.stop();
    thread_.wait();

    if (!thread_.openCamera(camIdx)) {
        QMessageBox::warning(this, "错误",
            QString("无法打开摄像头 %1。").arg(camIdx));
        return;
    }

    paused_ = false;
    pauseBtn_->setText("暂停");
    thread_.start();
    statusBar()->showMessage(QString("已切换到摄像头 %1").arg(camIdx), 2000);
}

void MainWindow::onConfChanged(int value)
{
    float threshold = value / 100.f;
    thread_.detector().setConfThreshold(threshold);
    confValueLabel_->setText(QString::number(threshold, 'f', 2));
}

void MainWindow::onIouChanged(int value)
{
    float threshold = value / 100.f;
    thread_.detector().setIouThreshold(threshold);
    iouValueLabel_->setText(QString::number(threshold, 'f', 2));
}

void MainWindow::onToggleRecord()
{
    if (thread_.isRecording()) {
        thread_.stopRecording();
        recordBtn_->setText("录制");
        statusBar()->showMessage("录制已停止", 3000);
    } else {
        QString defaultName = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".mp4";
        QString path = QFileDialog::getSaveFileName(this, "保存录制", defaultName,
            "视频 (*.mp4 *.avi)");
        if (path.isEmpty()) return;

        // Get source dimensions
        int width = 640, height = 480;
        if (thread_.isVideoSource()) {
            // Will use actual frame size in the thread
        }

        thread_.startRecording(path.toStdString(), 30.0, width, height);
        if (thread_.isRecording()) {
            recordBtn_->setText("停止录制");
            statusBar()->showMessage("录制中...", 3000);
        } else {
            QMessageBox::warning(this, "错误", "无法创建录制文件。");
        }
    }
}

void MainWindow::onNetworkCamera()
{
    bool ok;
    QString url = QInputDialog::getText(this, "网络摄像头",
        "输入 RTSP/HTTP 视频 URL:", QLineEdit::Normal, "rtsp://", &ok);
    if (!ok || url.trimmed().isEmpty()) return;

    setCursor(Qt::WaitCursor);

    thread_.stop();
    thread_.wait();

    if (!thread_.openVideo(url.toStdString())) {
        setCursor(Qt::ArrowCursor);
        QMessageBox::warning(this, "错误",
            "无法打开网络视频流。\n请检查 URL 是否正确以及网络连接。");
        return;
    }

    setCursor(Qt::ArrowCursor);
    paused_ = false;
    pauseBtn_->setText("暂停");
    thread_.start();
    statusBar()->showMessage("网络流: " + url, 3000);
}

void MainWindow::onSwitchModel()
{
    QString dir = QFileInfo(currentModelPath_).absolutePath();
    QString path = QFileDialog::getOpenFileName(this,
        "选择 ONNX 模型", dir, "ONNX 模型 (*.onnx)");
    if (path.isEmpty()) return;

    thread_.stop();
    thread_.wait();
    thread_.resetTracker();

    if (!thread_.detector().loadModel(path.toStdWString())) {
        QMessageBox::critical(this, "错误", "无法加载模型: " + path);
        thread_.start();
        return;
    }

    currentModelPath_ = path;
    deviceLabel_->setText(thread_.detector().isGpuEnabled() ? "GPU (CUDA)" : "CPU");
    paused_ = false;
    pauseBtn_->setText("暂停");
    thread_.start();
    statusBar()->showMessage("模型已切换: " + path, 3000);
}

void MainWindow::onClassFilter()
{
    ClassFilterDialog dlg(thread_.detector().enabledClasses(), this);
    if (dlg.exec() == QDialog::Accepted) {
        QSet<int> selected = dlg.selectedClasses();
        thread_.detector().setEnabledClasses(selected);
        enabledClasses_ = selected;
    }
}

void MainWindow::onToggleTracking(bool checked)
{
    thread_.setTrackingEnabled(checked);
    if (!checked) {
        thread_.stop();
        thread_.wait();
        thread_.resetTracker();
        thread_.start();
    }
    trackingBtn_->setText(checked ? "关闭追踪" : "目标追踪");
    statusBar()->showMessage(checked ? "目标追踪已启用" : "目标追踪已关闭", 2000);
}

void MainWindow::restartThread()
{
    thread_.stop();
    thread_.wait();
    thread_.start();
}
