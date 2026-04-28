#include "mainwindow.h"
#include "classfilterdialog.h"
#include "lang.h"
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
#include <QHeaderView>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QMenuBar>
#include <QMenu>
#include <QMimeData>
#include <QUrl>
#include <QProgressDialog>
#include <opencv2/core/version.hpp>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowIcon(QIcon("app.ico"));
    resize(960, 720);
    setAcceptDrops(true);

    setupUI();
    loadSettings();

    // Load model
    QSettings initSettings("DetectionAI", "YOLODetector");
    QString modelPath = initSettings.value("modelPath", "yolo11n.onnx").toString();
    if (!thread_.detector().loadModel(modelPath.toStdWString())) {
        modelPath = QFileDialog::getOpenFileName(this,
            Lang::s("select_model"), "", Lang::s("model_filter"));
        if (modelPath.isEmpty() || !thread_.detector().loadModel(modelPath.toStdWString())) {
            QMessageBox::critical(this, Lang::s("error"), Lang::s("model_load_fail"));
            return;
        }
    }
    currentModelPath_ = modelPath;
    addRecentModel(modelPath);

    deviceLabel_->setText(thread_.detector().isGpuEnabled()
        ? Lang::s("device_gpu") : Lang::s("device_cpu"));

    // Open default camera
    int cam = cameraCombo_->currentData().toInt();
    if (!thread_.openCamera(cam)) {
        QMessageBox::warning(this, Lang::s("cam_warn"), Lang::s("cam_open_fail"));
    }

    // Connect signals
    connect(&thread_, &InferenceThread::frameReady,
            this, qOverload<const QImage&, int, float, float, const QMap<int,int>&>(
                &MainWindow::onFrameReady));
    connect(&thread_, &InferenceThread::inputLost,
            this, &MainWindow::onInputLost);

    thread_.start();
}

MainWindow::~MainWindow()
{
    thread_.stop();
    thread_.wait();
}

void MainWindow::enumerateCameras()
{
    cameraCombo_->blockSignals(true);
    cameraCombo_->clear();
    auto oldLevel = cv::utils::logging::getLogLevel();
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    for (int i = 0; i < 10; i++) {
        cv::VideoCapture cap(i);
        if (cap.isOpened()) {
            cameraCombo_->addItem(Lang::s("camera").arg(i), i);
            cap.release();
        }
    }
    cv::utils::logging::setLogLevel(oldLevel);
    if (cameraCombo_->count() == 0)
        cameraCombo_->addItem(Lang::s("camera").arg(0), 0);
    cameraCombo_->blockSignals(false);
}

void MainWindow::setupUI()
{
    // --- Toolbar ---
    QToolBar* toolbar = addToolBar("Controls");
    toolbar->setMovable(false);

    pauseBtn_       = new QPushButton(Lang::s("pause"));
    screenshotBtn_  = new QPushButton(Lang::s("screenshot"));
    recordBtn_      = new QPushButton(Lang::s("record"));
    exportBtn_      = new QPushButton(Lang::s("export_btn"));
    videoBtn_       = new QPushButton(Lang::s("open_video"));
    networkCamBtn_  = new QPushButton(Lang::s("network_cam"));
    loopBtn_        = new QPushButton(Lang::s("loop"));
    switchModelBtn_ = new QPushButton(Lang::s("switch_model"));
    recentModelBtn_ = new QPushButton(Lang::s("recent_models"));
    classFilterBtn_ = new QPushButton(Lang::s("class_filter"));
    trackingBtn_    = new QPushButton(Lang::s("tracking_off"));
    langBtn_        = new QPushButton(Lang::s("lang_toggle"));

    loopBtn_->setCheckable(true);
    trackingBtn_->setCheckable(true);

    pauseBtn_->setToolTip(Lang::s("tip_pause"));
    screenshotBtn_->setToolTip(Lang::s("tip_screenshot"));
    recordBtn_->setToolTip(Lang::s("tip_record"));
    exportBtn_->setToolTip(Lang::s("tip_export"));
    videoBtn_->setToolTip(Lang::s("tip_open_video"));
    networkCamBtn_->setToolTip(Lang::s("tip_network"));
    loopBtn_->setToolTip(Lang::s("tip_loop"));
    switchModelBtn_->setToolTip(Lang::s("tip_model"));
    classFilterBtn_->setToolTip(Lang::s("tip_filter"));
    trackingBtn_->setToolTip(Lang::s("tip_tracking"));
    langBtn_->setToolTip(Lang::s("tip_lang"));

    cameraCombo_ = new QComboBox;
    enumerateCameras();
    cameraCombo_->setCurrentIndex(0);

    toolbar->addWidget(pauseBtn_);
    toolbar->addSeparator();
    toolbar->addWidget(screenshotBtn_);
    toolbar->addWidget(recordBtn_);
    toolbar->addWidget(exportBtn_);
    toolbar->addSeparator();
    toolbar->addWidget(videoBtn_);
    toolbar->addWidget(networkCamBtn_);
    toolbar->addWidget(loopBtn_);
    toolbar->addSeparator();
    toolbar->addWidget(switchModelBtn_);
    toolbar->addWidget(recentModelBtn_);
    toolbar->addSeparator();
    toolbar->addWidget(classFilterBtn_);
    toolbar->addWidget(trackingBtn_);
    toolbar->addSeparator();
    toolbar->addWidget(langBtn_);
    toolbar->addSeparator();
    toolbar->addWidget(new QLabel(Lang::s("input_source")));
    toolbar->addWidget(cameraCombo_);

    connect(pauseBtn_, &QPushButton::clicked, this, &MainWindow::onTogglePause);
    connect(screenshotBtn_, &QPushButton::clicked, this, &MainWindow::onScreenshot);
    connect(recordBtn_, &QPushButton::clicked, this, &MainWindow::onToggleRecord);
    connect(exportBtn_, &QPushButton::clicked, this, &MainWindow::onExport);
    connect(videoBtn_, &QPushButton::clicked, this, &MainWindow::onOpenVideo);
    connect(networkCamBtn_, &QPushButton::clicked, this, &MainWindow::onNetworkCamera);
    connect(loopBtn_, &QPushButton::toggled, this, &MainWindow::onToggleLoop);
    connect(switchModelBtn_, &QPushButton::clicked, this, &MainWindow::onSwitchModel);
    connect(recentModelBtn_, &QPushButton::clicked, this, &MainWindow::onRecentModel);
    connect(classFilterBtn_, &QPushButton::clicked, this, &MainWindow::onClassFilter);
    connect(trackingBtn_, &QPushButton::toggled, this, &MainWindow::onToggleTracking);
    connect(langBtn_, &QPushButton::clicked, this, &MainWindow::onToggleLanguage);
    connect(cameraCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onCameraChanged);

    // --- Sliders ---
    QWidget* sliderWidget = new QWidget;
    QHBoxLayout* sliderLayout = new QHBoxLayout(sliderWidget);
    sliderLayout->setContentsMargins(8, 4, 8, 4);

    sliderLayout->addWidget(new QLabel(Lang::s("confidence")));
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

    sliderLayout->addWidget(new QLabel(Lang::s("iou")));
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
    fpsLabel_    = new QLabel("FPS: --");
    detLabel_    = new QLabel(Lang::s("det_count").arg(0));
    inferLabel_  = new QLabel(Lang::s("infer_ms").arg(0));
    deviceLabel_ = new QLabel(Lang::s("device_cpu"));

    statusBar()->addWidget(fpsLabel_);
    statusBar()->addWidget(detLabel_);
    statusBar()->addWidget(inferLabel_);
    statusBar()->addPermanentWidget(deviceLabel_);

    // --- Help menu (About) ---
    QMenu* helpMenu = menuBar()->addMenu(Lang::s("about"));
    helpMenu->addAction(Lang::s("about"), this, &MainWindow::onAbout);

    // --- Statistics dock ---
    statsDock_ = new QDockWidget(Lang::s("stats_title"), this);
    auto* dockWidget = new QWidget;
    auto* dockLayout = new QVBoxLayout(dockWidget);
    dockLayout->setContentsMargins(0, 0, 0, 0);

    statsTable_ = new QTableWidget(0, 2, dockWidget);
    statsTable_->setHorizontalHeaderLabels(
        {Lang::s("stats_class"), Lang::s("stats_count")});
    statsTable_->horizontalHeader()->setStretchLastSection(true);
    statsTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    statsTable_->setSelectionBehavior(QAbstractItemView::SelectRows);

    clearStatsBtn_ = new QPushButton(Lang::s("stats_clear"));
    dockLayout->addWidget(statsTable_);
    dockLayout->addWidget(clearStatsBtn_);
    statsDock_->setWidget(dockWidget);
    addDockWidget(Qt::RightDockWidgetArea, statsDock_);

    connect(clearStatsBtn_, &QPushButton::clicked, this, &MainWindow::onClearStats);
}

void MainWindow::loadSettings()
{
    QSettings settings("DetectionAI", "YOLODetector");
    float conf = settings.value("confidence", 0.25).toFloat();
    float iou = settings.value("iou", 0.45).toFloat();
    int cam = settings.value("camera", 0).toInt();
    QSize winSize = settings.value("windowSize", QSize(960, 720)).toSize();
    QPoint winPos = settings.value("windowPos", QPoint()).toPoint();
    bool tracking = settings.value("tracking", false).toBool();
    bool loop = settings.value("loop", false).toBool();
    int lang = settings.value("language", 0).toInt();

    Lang::setLanguage(static_cast<Lang::Language>(lang));

    confSlider_->setValue((int)(conf * 100));
    iouSlider_->setValue((int)(iou * 100));
    cameraCombo_->setCurrentIndex(std::min(cam, cameraCombo_->count() - 1));
    resize(winSize);
    if (!winPos.isNull()) move(winPos);

    trackingBtn_->setChecked(tracking);
    thread_.setTrackingEnabled(tracking);
    loopBtn_->setChecked(loop);
    thread_.setLoopEnabled(loop);

    QList<QVariant> classList = settings.value("enabledClasses").toList();
    QSet<int> classes;
    for (const auto& v : std::as_const(classList)) classes.insert(v.toInt());
    thread_.detector().setEnabledClasses(classes);
    enabledClasses_ = classes;

    thread_.detector().setConfThreshold(conf);
    thread_.detector().setIouThreshold(iou);

    recentModels_ = settings.value("recentModels").toStringList();

    if (settings.value("statsVisible", true).toBool())
        statsDock_->show();
    else
        statsDock_->hide();
}

void MainWindow::saveSettings()
{
    QSettings settings("DetectionAI", "YOLODetector");
    settings.setValue("confidence", thread_.detector().confThreshold());
    settings.setValue("iou", thread_.detector().iouThreshold());
    settings.setValue("camera", cameraCombo_->currentIndex());
    settings.setValue("windowSize", size());
    settings.setValue("windowPos", pos());
    settings.setValue("modelPath", currentModelPath_);
    settings.setValue("tracking", thread_.isTrackingEnabled());
    settings.setValue("loop", thread_.isLoopEnabled());
    settings.setValue("language", static_cast<int>(Lang::language()));
    settings.setValue("statsVisible", statsDock_->isVisible());
    settings.setValue("recentModels", recentModels_);

    QList<QVariant> classList;
    for (int id : std::as_const(enabledClasses_)) classList.append(id);
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
    case Qt::Key_L:
        loopBtn_->toggle();
        break;
    case Qt::Key_E:
        onExport();
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

void MainWindow::dragEnterEvent(QDragEnterEvent* event)
{
    if (event->mimeData()->hasUrls()) {
        const auto urls = event->mimeData()->urls();
        if (!urls.isEmpty()) {
            QString suffix = QFileInfo(urls.first().toLocalFile()).suffix().toLower();
            if (suffix == "onnx" ||
                QStringList{"mp4","avi","mkv","mov","wmv"}.contains(suffix))
                event->acceptProposedAction();
        }
    }
}

void MainWindow::dropEvent(QDropEvent* event)
{
    const auto urls = event->mimeData()->urls();
    if (urls.isEmpty()) return;
    QString path = urls.first().toLocalFile();
    QString suffix = QFileInfo(path).suffix().toLower();

    if (suffix == "onnx") {
        loadModelFile(path);
    } else {
        openVideoFile(path);
    }
}

// --- Slots ---

void MainWindow::onFrameReady(const QImage& image, int detCount, float fps,
                               float inferMs, const QMap<int,int>& classCounts)
{
    if (image.isNull()) return;

    lastFrame_ = image;
    videoLabel_->setPixmap(QPixmap::fromImage(image).scaled(
        videoLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

    fpsLabel_->setText(Lang::s("fps").arg(fps, 0, 'f', 1));
    detLabel_->setText(Lang::s("det_count").arg(detCount));
    inferLabel_->setText(Lang::s("infer_ms").arg(inferMs, 0, 'f', 1));

    for (auto it = classCounts.begin(); it != classCounts.end(); ++it)
        classStats_[it.key()] += it.value();

    if (statsDock_->isVisible()) {
        statsTable_->setRowCount(classStats_.size());
        int row = 0;
        for (auto it = classStats_.constBegin(); it != classStats_.constEnd(); ++it) {
            statsTable_->setItem(row, 0,
                new QTableWidgetItem(QString::fromStdString(YOLODetector::CLASS_NAMES[it.key()])));
            statsTable_->setItem(row, 1, new QTableWidgetItem(QString::number(it.value())));
            row++;
        }
    }
}

void MainWindow::onInputLost(const QString& msg)
{
    statusBar()->showMessage(msg, 5000);
}

void MainWindow::onTogglePause()
{
    paused_ = !paused_;
    thread_.setPaused(paused_);
    pauseBtn_->setText(paused_ ? Lang::s("resume") : Lang::s("pause"));
}

void MainWindow::onScreenshot()
{
    if (lastFrame_.isNull()) return;

    QString defaultName = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".png";
    QString path = QFileDialog::getSaveFileName(this, Lang::s("save_screenshot"),
        defaultName, Lang::s("image_filter"));
    if (path.isEmpty()) return;

    if (lastFrame_.save(path))
        statusBar()->showMessage(Lang::s("screenshot_saved") + path, 3000);
    else
        QMessageBox::warning(this, Lang::s("error"), Lang::s("screenshot_fail"));
}

void MainWindow::onOpenVideo()
{
    QString path = QFileDialog::getOpenFileName(this, Lang::s("open_video_title"),
        "", Lang::s("video_filter"));
    if (path.isEmpty()) return;
    openVideoFile(path);
}

void MainWindow::openVideoFile(const QString& path)
{
    thread_.stop();
    thread_.wait();

    if (!thread_.openVideo(path.toStdString())) {
        QMessageBox::warning(this, Lang::s("error"), Lang::s("video_open_fail"));
        return;
    }

    paused_ = false;
    pauseBtn_->setText(Lang::s("pause"));
    thread_.start();
    statusBar()->showMessage(Lang::s("video_opened") + path, 3000);
}

void MainWindow::onCameraChanged(int index)
{
    if (index < 0) return;
    int camIdx = cameraCombo_->itemData(index).toInt();

    thread_.stop();
    thread_.wait();

    if (!thread_.openCamera(camIdx)) {
        QMessageBox::warning(this, Lang::s("error"),
            Lang::s("cam_switch_fail").arg(camIdx));
        return;
    }

    paused_ = false;
    pauseBtn_->setText(Lang::s("pause"));
    thread_.start();
    statusBar()->showMessage(Lang::s("cam_switched").arg(camIdx), 2000);
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
        recordBtn_->setText(Lang::s("record"));
        statusBar()->showMessage(Lang::s("recording_stopped"), 3000);
    } else {
        QString defaultName = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".mp4";
        QString path = QFileDialog::getSaveFileName(this, Lang::s("save_recording"),
            defaultName, Lang::s("video_save_filter"));
        if (path.isEmpty()) return;

        QSize sz = thread_.frameSize();
        int width = sz.width() > 0 ? sz.width() : 640;
        int height = sz.height() > 0 ? sz.height() : 480;

        thread_.startRecording(path.toStdString(), 30.0, width, height);
        if (thread_.isRecording()) {
            recordBtn_->setText(Lang::s("stop_record"));
            statusBar()->showMessage(Lang::s("recording"), 3000);
        } else {
            QMessageBox::warning(this, Lang::s("error"), Lang::s("recording_fail"));
        }
    }
}

void MainWindow::onNetworkCamera()
{
    bool ok;
    QString url = QInputDialog::getText(this, Lang::s("network_title"),
        Lang::s("network_prompt"), QLineEdit::Normal, "rtsp://", &ok);
    if (!ok || url.trimmed().isEmpty()) return;

    setCursor(Qt::WaitCursor);

    thread_.stop();
    thread_.wait();

    if (!thread_.openVideo(url.toStdString())) {
        setCursor(Qt::ArrowCursor);
        QMessageBox::warning(this, Lang::s("error"), Lang::s("network_fail"));
        return;
    }

    setCursor(Qt::ArrowCursor);
    paused_ = false;
    pauseBtn_->setText(Lang::s("pause"));
    thread_.start();
    statusBar()->showMessage(Lang::s("network_opened") + url, 3000);
}

void MainWindow::onSwitchModel()
{
    QString dir = QFileInfo(currentModelPath_).absolutePath();
    QString path = QFileDialog::getOpenFileName(this,
        Lang::s("select_model"), dir, Lang::s("model_filter"));
    if (path.isEmpty()) return;
    loadModelFile(path);
}

void MainWindow::loadModelFile(const QString& path)
{
    thread_.stop();
    thread_.wait();
    thread_.resetTracker();

    if (!thread_.detector().loadModel(path.toStdWString())) {
        QMessageBox::critical(this, Lang::s("error"),
            Lang::s("model_switch_fail") + path);
        thread_.start();
        return;
    }

    currentModelPath_ = path;
    addRecentModel(path);
    deviceLabel_->setText(thread_.detector().isGpuEnabled()
        ? Lang::s("device_gpu") : Lang::s("device_cpu"));
    paused_ = false;
    pauseBtn_->setText(Lang::s("pause"));
    thread_.start();
    statusBar()->showMessage(Lang::s("model_switched") + path, 3000);
}

void MainWindow::onRecentModel()
{
    if (recentModels_.isEmpty()) return;

    QMenu menu;
    for (const auto& m : std::as_const(recentModels_))
        menu.addAction(QFileInfo(m).fileName(), [this, m]() { loadModelFile(m); });
    menu.exec(recentModelBtn_->mapToGlobal(QPoint(0, recentModelBtn_->height())));
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
    trackingBtn_->setText(checked ? Lang::s("tracking_on") : Lang::s("tracking_off"));
    statusBar()->showMessage(
        checked ? Lang::s("tracking_enabled") : Lang::s("tracking_disabled"), 2000);
}

void MainWindow::onToggleLoop(bool checked)
{
    thread_.setLoopEnabled(checked);
    statusBar()->showMessage(
        checked ? Lang::s("loop_enabled") : Lang::s("loop_disabled"), 2000);
}

void MainWindow::onExport()
{
    auto dets = thread_.lastDetections();
    if (dets.empty()) {
        QMessageBox::information(this, Lang::s("export_title"), Lang::s("export_no_data"));
        return;
    }

    QString path = QFileDialog::getSaveFileName(this, Lang::s("export_title"),
        QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"),
        Lang::s("export_filter"));
    if (path.isEmpty()) return;

    bool success = false;
    if (path.endsWith(".csv", Qt::CaseInsensitive)) {
        QFile file(path);
        if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&file);
            out << "class,classId,confidence,x,y,width,height\n";
            for (const auto& d : dets) {
                out << QString::fromStdString(YOLODetector::CLASS_NAMES[d.classId]) << ","
                    << d.classId << ","
                    << QString::number(d.confidence, 'f', 4) << ","
                    << d.bbox.x << "," << d.bbox.y << ","
                    << d.bbox.width << "," << d.bbox.height << "\n";
            }
            success = true;
        }
    } else {
        QJsonObject root;
        root["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);
        QJsonArray arr;
        for (const auto& d : dets) {
            QJsonObject obj;
            obj["class"] = QString::fromStdString(YOLODetector::CLASS_NAMES[d.classId]);
            obj["classId"] = d.classId;
            obj["confidence"] = qRound(d.confidence * 10000) / 10000.0;
            QJsonArray bbox;
            bbox.append(d.bbox.x);
            bbox.append(d.bbox.y);
            bbox.append(d.bbox.width);
            bbox.append(d.bbox.height);
            obj["bbox"] = bbox;
            arr.append(obj);
        }
        root["detections"] = arr;

        QFile file(path);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
            success = true;
        }
    }

    if (success)
        statusBar()->showMessage(Lang::s("export_done") + path, 3000);
    else
        QMessageBox::warning(this, Lang::s("error"), Lang::s("export_fail"));
}

void MainWindow::onToggleLanguage()
{
    Lang::setLanguage(Lang::language() == Lang::Chinese ? Lang::English : Lang::Chinese);
    refreshUIText();
}

void MainWindow::onAbout()
{
    QString cvVer = QString("%1.%2.%3")
        .arg(CV_VERSION_MAJOR).arg(CV_VERSION_MINOR).arg(CV_VERSION_REVISION);
    QMessageBox::about(this, Lang::s("about"),
        Lang::s("about_text").arg(qVersion()).arg(cvVer));
}

void MainWindow::onClearStats()
{
    classStats_.clear();
    statsTable_->setRowCount(0);
}

void MainWindow::refreshUIText()
{
    setWindowTitle(Lang::s("app_title"));
    pauseBtn_->setText(paused_ ? Lang::s("resume") : Lang::s("pause"));
    screenshotBtn_->setText(Lang::s("screenshot"));
    recordBtn_->setText(thread_.isRecording() ? Lang::s("stop_record") : Lang::s("record"));
    exportBtn_->setText(Lang::s("export_btn"));
    videoBtn_->setText(Lang::s("open_video"));
    networkCamBtn_->setText(Lang::s("network_cam"));
    loopBtn_->setText(Lang::s("loop"));
    switchModelBtn_->setText(Lang::s("switch_model"));
    recentModelBtn_->setText(Lang::s("recent_models"));
    classFilterBtn_->setText(Lang::s("class_filter"));
    trackingBtn_->setText(trackingBtn_->isChecked() ? Lang::s("tracking_on") : Lang::s("tracking_off"));
    langBtn_->setText(Lang::s("lang_toggle"));
    clearStatsBtn_->setText(Lang::s("stats_clear"));
    deviceLabel_->setText(thread_.detector().isGpuEnabled()
        ? Lang::s("device_gpu") : Lang::s("device_cpu"));

    pauseBtn_->setToolTip(Lang::s("tip_pause"));
    screenshotBtn_->setToolTip(Lang::s("tip_screenshot"));
    recordBtn_->setToolTip(Lang::s("tip_record"));
    exportBtn_->setToolTip(Lang::s("tip_export"));
    videoBtn_->setToolTip(Lang::s("tip_open_video"));
    networkCamBtn_->setToolTip(Lang::s("tip_network"));
    loopBtn_->setToolTip(Lang::s("tip_loop"));
    switchModelBtn_->setToolTip(Lang::s("tip_model"));
    classFilterBtn_->setToolTip(Lang::s("tip_filter"));
    trackingBtn_->setToolTip(Lang::s("tip_tracking"));
    langBtn_->setToolTip(Lang::s("tip_lang"));

    statsDock_->setWindowTitle(Lang::s("stats_title"));
    statsTable_->setHorizontalHeaderLabels(
        {Lang::s("stats_class"), Lang::s("stats_count")});

    // Refresh camera combo
    int curCam = cameraCombo_->currentData().toInt();
    cameraCombo_->blockSignals(true);
    cameraCombo_->clear();
    for (int i = 0; i < 10; i++) {
        // Re-add previously found cameras (we can't re-enumerate here easily)
        cameraCombo_->addItem(Lang::s("camera").arg(i), i);
    }
    // Find the saved camera
    for (int i = 0; i < cameraCombo_->count(); i++) {
        if (cameraCombo_->itemData(i).toInt() == curCam) {
            cameraCombo_->setCurrentIndex(i);
            break;
        }
    }
    cameraCombo_->blockSignals(false);

    // Refresh help menu
    if (auto* menuBar = this->menuBar()) {
        auto actions = menuBar->actions();
        for (auto* a : std::as_const(actions)) {
            if (a->menu()) {
                a->menu()->setTitle(Lang::s("about"));
                for (auto* sub : a->menu()->actions())
                    sub->setText(Lang::s("about"));
            }
        }
    }
}

void MainWindow::addRecentModel(const QString& path)
{
    recentModels_.removeAll(path);
    recentModels_.prepend(path);
    while (recentModels_.size() > 5)
        recentModels_.removeLast();
}
