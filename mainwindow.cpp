#include "mainwindow.h"
#include "classfilterdialog.h"
#include "calibrationdialog.h"
#include "stereoettingsdialog.h"
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
#include <algorithm>
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
            Lang::s("select_model"), "", Lang::s("model_filter"),
            nullptr, QFileDialog::DontUseNativeDialog);
        if (modelPath.isEmpty() || !thread_.detector().loadModel(modelPath.toStdWString())) {
            QMessageBox::critical(this, Lang::s("error"), Lang::s("model_load_fail"));
            return;
        }
    }
    currentModelPath_ = modelPath;
    addRecentModel(modelPath);

    updateModelTypeUI();

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
    connect(&thread_, &InferenceThread::trackingStatsUpdated,
            this, &MainWindow::onTrackingStatsUpdated);
    connect(&thread_, &InferenceThread::crossingStatsUpdated,
            this, &MainWindow::onCrossingStatsUpdated);
    connect(&thread_, &InferenceThread::poseDataUpdated,
            this, &MainWindow::onPoseDataUpdated);
    connect(&thread_, &InferenceThread::depthMapReady,
            this, &MainWindow::onDepthMapReady);

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
    trajectoryBtn_  = new QPushButton(Lang::s("trajectory_off"));
    speedBtn_       = new QPushButton(Lang::s("speed_off"));
    skeletonBtn_    = new QPushButton(Lang::s("skeleton_off"));
    stereoBtn_      = new QPushButton(Lang::s("stereo_off"));
    calibrateBtn_   = new QPushButton(Lang::s("calibration"));
    depthOverlayBtn_ = new QPushButton(Lang::s("depth_overlay_off"));
    stereoSettingsBtn_ = new QPushButton(Lang::s("stereo_settings"));
    countLineBtn_   = new QPushButton(Lang::s("draw_line"));
    clearLineBtn_   = new QPushButton(Lang::s("clear_line"));
    langBtn_        = new QPushButton(Lang::s("lang_toggle"));

    loopBtn_->setCheckable(true);
    trackingBtn_->setCheckable(true);
    trajectoryBtn_->setCheckable(true);
    speedBtn_->setCheckable(true);
    skeletonBtn_->setCheckable(true);
    skeletonBtn_->setChecked(true);
    stereoBtn_->setCheckable(true);
    depthOverlayBtn_->setCheckable(true);

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
    trajectoryBtn_->setToolTip(Lang::s("tip_trajectory"));
    speedBtn_->setToolTip(Lang::s("tip_speed"));
    skeletonBtn_->setToolTip(Lang::s("tip_skeleton"));
    stereoBtn_->setToolTip(Lang::s("tip_stereo"));
    calibrateBtn_->setToolTip(Lang::s("tip_calibrate"));
    depthOverlayBtn_->setToolTip(Lang::s("tip_depth_overlay"));
    countLineBtn_->setToolTip(Lang::s("tip_draw_line"));
    clearLineBtn_->setToolTip(Lang::s("tip_clear_line"));
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
    toolbar->addWidget(trajectoryBtn_);
    toolbar->addWidget(speedBtn_);
    toolbar->addWidget(skeletonBtn_);
    toolbar->addWidget(stereoBtn_);
    toolbar->addWidget(calibrateBtn_);
    toolbar->addWidget(depthOverlayBtn_);
    toolbar->addWidget(stereoSettingsBtn_);
    toolbar->addWidget(countLineBtn_);
    toolbar->addWidget(clearLineBtn_);
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
    connect(trajectoryBtn_, &QPushButton::toggled, this, &MainWindow::onToggleTrajectory);
    connect(speedBtn_, &QPushButton::toggled, this, &MainWindow::onToggleSpeed);
    connect(skeletonBtn_, &QPushButton::toggled, this, &MainWindow::onToggleSkeleton);
    connect(stereoBtn_, &QPushButton::toggled, this, &MainWindow::onToggleStereo);
    connect(calibrateBtn_, &QPushButton::clicked, this, &MainWindow::onCalibrate);
    connect(depthOverlayBtn_, &QPushButton::toggled, this, &MainWindow::onToggleDepthOverlay);
    connect(stereoSettingsBtn_, &QPushButton::clicked, this, &MainWindow::onStereoSettings);
    connect(countLineBtn_, &QPushButton::clicked, this, &MainWindow::onDrawCountingLine);
    connect(clearLineBtn_, &QPushButton::clicked, this, &MainWindow::onClearCountingLine);
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
    videoLabel_->installEventFilter(this);

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

    statsTable_ = new QTableWidget(0, 3, dockWidget);
    statsTable_->setHorizontalHeaderLabels(
        {Lang::s("stats_class"), Lang::s("stats_count"), Lang::s("stats_unique")});
    statsTable_->horizontalHeader()->setStretchLastSection(true);
    statsTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    statsTable_->setSelectionBehavior(QAbstractItemView::SelectRows);

    totalUniqueLabel_ = new QLabel(Lang::s("stats_total_unique").arg(0));

    auto* statsBtnLayout = new QHBoxLayout;
    clearStatsBtn_ = new QPushButton(Lang::s("stats_clear"));
    resetCountsBtn_ = new QPushButton(Lang::s("reset_counts"));
    statsBtnLayout->addWidget(clearStatsBtn_);
    statsBtnLayout->addWidget(resetCountsBtn_);

    dockLayout->addWidget(statsTable_);
    dockLayout->addWidget(totalUniqueLabel_);
    dockLayout->addLayout(statsBtnLayout);
    statsDock_->setWidget(dockWidget);
    addDockWidget(Qt::RightDockWidgetArea, statsDock_);

    connect(clearStatsBtn_, &QPushButton::clicked, this, &MainWindow::onClearStats);
    connect(resetCountsBtn_, &QPushButton::clicked, this, [this]() {
        thread_.resetTrackCounts();
        uniqueCounts_.clear();
        totalUniqueLabel_->setText(Lang::s("stats_total_unique").arg(0));
        for (int r = 0; r < statsTable_->rowCount(); r++)
            statsTable_->setItem(r, 2, new QTableWidgetItem("0"));
    });

    // --- Counting dock ---
    countingDock_ = new QDockWidget(Lang::s("crossing_count"), this);
    auto* countWidget = new QWidget;
    auto* countLayout = new QVBoxLayout(countWidget);
    countLayout->setContentsMargins(0, 0, 0, 0);

    countingTable_ = new QTableWidget(0, 4, countWidget);
    countingTable_->setHorizontalHeaderLabels(
        {Lang::s("stats_class"), Lang::s("forward"), Lang::s("reverse"), Lang::s("total")});
    countingTable_->horizontalHeader()->setStretchLastSection(true);
    countingTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    countingTable_->setSelectionBehavior(QAbstractItemView::SelectRows);

    auto* countBtnLayout = new QHBoxLayout;
    clearCrossingBtn_ = new QPushButton(Lang::s("clear_line"));
    countBtnLayout->addWidget(clearCrossingBtn_);

    countLayout->addWidget(countingTable_);
    countLayout->addLayout(countBtnLayout);
    countingDock_->setWidget(countWidget);
    addDockWidget(Qt::RightDockWidgetArea, countingDock_);
    tabifyDockWidget(statsDock_, countingDock_);

    connect(clearCrossingBtn_, &QPushButton::clicked, this, [this]() {
        thread_.clearCountingLine();
        thread_.resetCrossingCounts();
        countingTable_->setRowCount(0);
    });

    // --- Pose data dock ---
    poseDock_ = new QDockWidget(Lang::s("pose_title"), this);
    auto* poseWidget = new QWidget;
    auto* poseLayout = new QVBoxLayout(poseWidget);
    poseLayout->setContentsMargins(0, 0, 0, 0);

    posePersonLabel_ = new QLabel("");
    poseLayout->addWidget(posePersonLabel_);

    poseTable_ = new QTableWidget(0, 3, poseWidget);
    poseTable_->setHorizontalHeaderLabels(
        {Lang::s("pose_keypoint"), Lang::s("pose_position"), Lang::s("pose_confidence")});
    poseTable_->horizontalHeader()->setStretchLastSection(true);
    poseTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    poseTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    poseLayout->addWidget(poseTable_);

    auto* kpConfLayout = new QHBoxLayout;
    kpConfLayout->addWidget(new QLabel(Lang::s("kp_conf_threshold")));
    kpConfSlider_ = new QSlider(Qt::Horizontal);
    kpConfSlider_->setRange(10, 95);
    kpConfSlider_->setValue(50);
    kpConfSlider_->setMinimumWidth(100);
    kpConfValueLabel_ = new QLabel("0.50");
    kpConfValueLabel_->setMinimumWidth(36);
    kpConfLayout->addWidget(kpConfSlider_);
    kpConfLayout->addWidget(kpConfValueLabel_);
    poseLayout->addLayout(kpConfLayout);

    connect(kpConfSlider_, &QSlider::valueChanged, this, [this](int value) {
        float threshold = value / 100.f;
        thread_.setKeypointConfThreshold(threshold);
        kpConfValueLabel_->setText(QString::number(threshold, 'f', 2));
    });

    poseDock_->setWidget(poseWidget);
    addDockWidget(Qt::RightDockWidgetArea, poseDock_);
    tabifyDockWidget(statsDock_, poseDock_);
    poseDock_->hide();
    skeletonBtn_->setVisible(false);

    // --- Depth data dock ---
    depthDock_ = new QDockWidget(Lang::s("depth_dock"), this);
    auto* depthWidget = new QWidget;
    auto* depthLayout = new QVBoxLayout(depthWidget);
    depthLayout->setContentsMargins(0, 0, 0, 0);

    depthTable_ = new QTableWidget(0, 4, depthWidget);
    depthTable_->setHorizontalHeaderLabels(
        {Lang::s("depth_track_id"), Lang::s("depth_class"),
         Lang::s("depth_dist"), Lang::s("depth_conf")});
    depthTable_->horizontalHeader()->setStretchLastSection(true);
    depthTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    depthTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    depthLayout->addWidget(depthTable_);

    depthDock_->setWidget(depthWidget);
    addDockWidget(Qt::RightDockWidgetArea, depthDock_);
    tabifyDockWidget(statsDock_, depthDock_);
    depthDock_->hide();

    // --- Point cloud dock ---
    pointCloudDock_ = new QDockWidget(Lang::s("point_cloud_dock"), this);
    auto* pcWidget = new QWidget;
    auto* pcLayout = new QVBoxLayout(pcWidget);
    pcLayout->setContentsMargins(0, 0, 0, 0);

    pointCloudLabel_ = new QLabel;
    pointCloudLabel_->setAlignment(Qt::AlignCenter);
    pointCloudLabel_->setStyleSheet("QLabel { background-color: #1a1a1a; }");
    pointCloudLabel_->setMinimumSize(300, 300);
    pcLayout->addWidget(pointCloudLabel_);

    pointCloudDock_->setWidget(pcWidget);
    addDockWidget(Qt::RightDockWidgetArea, pointCloudDock_);
    tabifyDockWidget(statsDock_, pointCloudDock_);
    pointCloudDock_->hide();

    // Stereo buttons hidden by default
    calibrateBtn_->setVisible(false);
    depthOverlayBtn_->setVisible(false);
    stereoSettingsBtn_->setVisible(false);
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
    bool trajectory = settings.value("trajectory", false).toBool();
    trajectoryBtn_->setChecked(trajectory);
    thread_.setTrajectoryEnabled(trajectory);
    bool speed = settings.value("speed", false).toBool();
    speedBtn_->setChecked(speed);
    thread_.setSpeedEnabled(speed);
    bool skeleton = settings.value("skeleton", true).toBool();
    skeletonBtn_->setChecked(skeleton);
    thread_.setSkeletonEnabled(skeleton);
    bool stereo = settings.value("stereoMode", false).toBool();
    stereoBtn_->setChecked(stereo);
    thread_.setStereoMode(stereo);
    bool depthOverlay = settings.value("depthOverlay", false).toBool();
    depthOverlayBtn_->setChecked(depthOverlay);
    thread_.setDepthOverlay(depthOverlay);
    float kpConf = settings.value("keypointConfThreshold", 0.5).toFloat();
    kpConfSlider_->setValue((int)(kpConf * 100));
    thread_.setKeypointConfThreshold(kpConf);
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

    // Stereo config
    stereoConfig_.hardware = static_cast<StereoHardware>(
        settings.value("stereoHardware", 0).toInt());
    stereoConfig_.leftCameraIndex = settings.value("stereoLeftCam", 0).toInt();
    stereoConfig_.rightCameraIndex = settings.value("stereoRightCam", 1).toInt();
    stereoConfig_.leftRTSPUrl = settings.value("stereoLeftRTSP", "").toString().toStdString();
    stereoConfig_.rightRTSPUrl = settings.value("stereoRightRTSP", "").toString().toStdString();
    stereoConfig_.targetWidth = settings.value("stereoWidth", 640).toInt();
    stereoConfig_.targetHeight = settings.value("stereoHeight", 480).toInt();

    sgbmParams_.blockSize = settings.value("sgbmBlockSize", 5).toInt();
    sgbmParams_.minDisparity = settings.value("sgbmMinDisp", 0).toInt();
    sgbmParams_.numDisparities = settings.value("sgbmNumDisp", 64).toInt();
    sgbmParams_.uniquenessRatio = settings.value("sgbmUniqueness", 10).toInt();
    sgbmParams_.speckleWindowSize = settings.value("sgbmSpeckleWin", 100).toInt();
    sgbmParams_.speckleRange = settings.value("sgbmSpeckleRange", 32).toInt();
    sgbmParams_.baselineMeters = settings.value("sgbmBaseline", 0.06).toFloat();
    sgbmParams_.focalLengthPixels = settings.value("sgbmFocal", 700.0).toFloat();
    thread_.setSGBMParams(sgbmParams_);

    lastCalibPath_ = settings.value("lastCalibPath", "").toString();
    if (!lastCalibPath_.isEmpty()) {
        StereoRectifier rectifier;
        if (rectifier.loadCalibration(lastCalibPath_.toStdString()))
            thread_.setStereoCalibration(rectifier.calibration());
    }

    if (settings.value("statsVisible", true).toBool())
        statsDock_->show();
    else
        statsDock_->hide();

    if (settings.value("countingDockVisible", false).toBool())
        countingDock_->show();
    else
        countingDock_->hide();

    if (settings.value("poseDockVisible", false).toBool())
        poseDock_->show();
    else
        poseDock_->hide();

    if (settings.value("depthDockVisible", false).toBool())
        depthDock_->show();
    else
        depthDock_->hide();

    if (settings.value("pointCloudDockVisible", false).toBool())
        pointCloudDock_->show();
    else
        pointCloudDock_->hide();
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
    settings.setValue("trajectory", thread_.isTrajectoryEnabled());
    settings.setValue("speed", thread_.isSpeedEnabled());
    settings.setValue("skeleton", thread_.isSkeletonEnabled());
    settings.setValue("keypointConfThreshold", thread_.keypointConfThreshold());
    settings.setValue("loop", thread_.isLoopEnabled());
    settings.setValue("language", static_cast<int>(Lang::language()));
    settings.setValue("stereoMode", thread_.isStereoMode());
    settings.setValue("depthOverlay", thread_.depthOverlayEnabled());
    settings.setValue("statsVisible", statsDock_->isVisible());
    settings.setValue("countingDockVisible", countingDock_->isVisible());
    settings.setValue("poseDockVisible", poseDock_->isVisible());
    settings.setValue("depthDockVisible", depthDock_->isVisible());
    settings.setValue("pointCloudDockVisible", pointCloudDock_->isVisible());
    settings.setValue("recentModels", recentModels_);

    settings.setValue("stereoHardware", static_cast<int>(stereoConfig_.hardware));
    settings.setValue("stereoLeftCam", stereoConfig_.leftCameraIndex);
    settings.setValue("stereoRightCam", stereoConfig_.rightCameraIndex);
    settings.setValue("stereoLeftRTSP", QString::fromStdString(stereoConfig_.leftRTSPUrl));
    settings.setValue("stereoRightRTSP", QString::fromStdString(stereoConfig_.rightRTSPUrl));
    settings.setValue("stereoWidth", stereoConfig_.targetWidth);
    settings.setValue("stereoHeight", stereoConfig_.targetHeight);

    settings.setValue("sgbmBlockSize", sgbmParams_.blockSize);
    settings.setValue("sgbmMinDisp", sgbmParams_.minDisparity);
    settings.setValue("sgbmNumDisp", sgbmParams_.numDisparities);
    settings.setValue("sgbmUniqueness", sgbmParams_.uniquenessRatio);
    settings.setValue("sgbmSpeckleWin", sgbmParams_.speckleWindowSize);
    settings.setValue("sgbmSpeckleRange", sgbmParams_.speckleRange);
    settings.setValue("sgbmBaseline", sgbmParams_.baselineMeters);
    settings.setValue("sgbmFocal", sgbmParams_.focalLengthPixels);
    settings.setValue("lastCalibPath", lastCalibPath_);

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
        if (event->modifiers() & Qt::ShiftModifier)
            speedBtn_->toggle();
        else
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
        if (event->modifiers() & Qt::ShiftModifier)
            trajectoryBtn_->toggle();
        else
            trackingBtn_->toggle();
        break;
    case Qt::Key_L:
        loopBtn_->toggle();
        break;
    case Qt::Key_C:
        onDrawCountingLine();
        break;
    case Qt::Key_K:
        if (thread_.detector().isPoseModel())
            skeletonBtn_->toggle();
        break;
    case Qt::Key_B:
        if (event->modifiers() & Qt::ShiftModifier)
            onCalibrate();
        else
            stereoBtn_->toggle();
        break;
    case Qt::Key_D:
        if (thread_.isStereoMode())
            depthOverlayBtn_->toggle();
        break;
    case Qt::Key_E:
        onExport();
        break;
    case Qt::Key_F11:
        isFullScreen() ? showNormal() : showFullScreen();
        break;
    case Qt::Key_Escape:
        if (drawMode_ != DrawMode::Idle) {
            drawMode_ = DrawMode::Idle;
            videoLabel_->setCursor(Qt::ArrowCursor);
            statusBar()->showMessage(Lang::s("draw_cancelled"), 3000);
        } else if (isFullScreen()) {
            showNormal();
        } else {
            close();
        }
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

QPoint MainWindow::widgetToFrameCoords(const QPoint& widgetPos) const
{
    QPixmap pm = videoLabel_->pixmap();
    if (pm.isNull()) return QPoint();

    int pw = pm.width(), ph = pm.height();
    int lw = videoLabel_->width(), lh = videoLabel_->height();
    int offsetX = (lw - pw) / 2;
    int offsetY = (lh - ph) / 2;

    int px = widgetPos.x() - offsetX;
    int py = widgetPos.y() - offsetY;
    if (px < 0 || py < 0 || px >= pw || py >= ph)
        return QPoint();

    QSize frameSz = thread_.frameSize();
    if (frameSz.isEmpty()) return QPoint();

    return QPoint(px * frameSz.width() / pw, py * frameSz.height() / ph);
}

bool MainWindow::eventFilter(QObject* watched, QEvent* event)
{
    if (watched == videoLabel_ && event->type() == QEvent::MouseButtonPress) {
        auto* me = static_cast<QMouseEvent*>(event);
        if (me->button() == Qt::LeftButton && drawMode_ != DrawMode::Idle) {
            QPoint framePt = widgetToFrameCoords(me->pos());
            if (framePt.isNull()) return true;

            if (drawMode_ == DrawMode::WaitingPt1) {
                drawPt1_ = cv::Point(framePt.x(), framePt.y());
                drawMode_ = DrawMode::WaitingPt2;
                statusBar()->showMessage(Lang::s("click_pt2"), 10000);
            } else if (drawMode_ == DrawMode::WaitingPt2) {
                cv::Point pt2(framePt.x(), framePt.y());
                drawMode_ = DrawMode::Idle;
                videoLabel_->setCursor(Qt::ArrowCursor);

                bool ok;
                QString label = QInputDialog::getText(this, Lang::s("counting_line"),
                    Lang::s("line_label_prompt"), QLineEdit::Normal, "", &ok);
                if (!ok) {
                    statusBar()->showMessage(Lang::s("draw_cancelled"), 3000);
                    return true;
                }

                CountingLine line;
                line.pt1 = drawPt1_;
                line.pt2 = pt2;
                line.label = label.toStdString();
                thread_.setCountingLine(line);
                statusBar()->showMessage(Lang::s("line_set"), 3000);
            }
            return true;
        }
    }
    return QMainWindow::eventFilter(watched, event);
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
        QSet<int> allClasses;
        for (auto it = classStats_.constBegin(); it != classStats_.constEnd(); ++it)
            allClasses.insert(it.key());
        for (auto it = uniqueCounts_.constBegin(); it != uniqueCounts_.constEnd(); ++it)
            allClasses.insert(it.key());

        statsTable_->setRowCount(allClasses.size());
        int row = 0;
        for (int cls : allClasses) {
            statsTable_->setItem(row, 0,
                new QTableWidgetItem(QString::fromStdString(YOLODetector::CLASS_NAMES[cls])));
            statsTable_->setItem(row, 1,
                new QTableWidgetItem(QString::number(classStats_.value(cls, 0))));
            statsTable_->setItem(row, 2,
                new QTableWidgetItem(QString::number(uniqueCounts_.value(cls, 0))));
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
        defaultName, Lang::s("image_filter"), nullptr, QFileDialog::DontUseNativeDialog);
    if (path.isEmpty()) return;

    if (lastFrame_.save(path))
        statusBar()->showMessage(Lang::s("screenshot_saved") + path, 3000);
    else
        QMessageBox::warning(this, Lang::s("error"), Lang::s("screenshot_fail"));
}

void MainWindow::onOpenVideo()
{
    QString path = QFileDialog::getOpenFileName(this, Lang::s("open_video_title"),
        "", Lang::s("video_filter"), nullptr, QFileDialog::DontUseNativeDialog);
    if (path.isEmpty()) return;
    openVideoFile(path);
}

void MainWindow::openVideoFile(const QString& path)
{
    thread_.stop();
    thread_.wait();

    if (!thread_.openVideo(path.toLocal8Bit().toStdString())) {
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
            defaultName, Lang::s("video_save_filter"), nullptr, QFileDialog::DontUseNativeDialog);
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
        Lang::s("select_model"), dir, Lang::s("model_filter"),
        nullptr, QFileDialog::DontUseNativeDialog);
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
    updateModelTypeUI();
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
        uniqueCounts_.clear();
        countingTable_->setRowCount(0);
    }
    trackingBtn_->setText(checked ? Lang::s("tracking_on") : Lang::s("tracking_off"));
    statusBar()->showMessage(
        checked ? Lang::s("tracking_enabled") : Lang::s("tracking_disabled"), 2000);
}

void MainWindow::onToggleTrajectory(bool checked)
{
    thread_.setTrajectoryEnabled(checked);
    trajectoryBtn_->setText(checked ? Lang::s("trajectory_on") : Lang::s("trajectory_off"));
    statusBar()->showMessage(
        checked ? Lang::s("trajectory_on") : Lang::s("trajectory_off"), 2000);
}

void MainWindow::onToggleSpeed(bool checked)
{
    thread_.setSpeedEnabled(checked);
    speedBtn_->setText(checked ? Lang::s("speed_on") : Lang::s("speed_off"));
    statusBar()->showMessage(
        checked ? Lang::s("speed_on") : Lang::s("speed_off"), 2000);
}

void MainWindow::onToggleSkeleton(bool checked)
{
    thread_.setSkeletonEnabled(checked);
    skeletonBtn_->setText(checked ? Lang::s("skeleton_on") : Lang::s("skeleton_off"));
}

void MainWindow::onToggleStereo(bool checked)
{
    if (checked) {
        // Open stereo source
        if (!thread_.openStereo(stereoConfig_)) {
            QMessageBox::warning(this, Lang::s("error"), Lang::s("stereo_open_fail"));
            stereoBtn_->setChecked(false);
            return;
        }
        thread_.setStereoMode(true);
        statusBar()->showMessage(Lang::s("stereo_connected"), 3000);
    } else {
        thread_.setStereoMode(false);
        thread_.openCamera(cameraCombo_->currentData().toInt());
        depthDock_->hide();
        pointCloudDock_->hide();
        statusBar()->showMessage(Lang::s("stereo_disconnected"), 3000);
    }

    stereoBtn_->setText(checked ? Lang::s("stereo_on") : Lang::s("stereo_off"));
    calibrateBtn_->setVisible(checked);
    depthOverlayBtn_->setVisible(checked);
    stereoSettingsBtn_->setVisible(checked);
    updateModelTypeUI();
}

void MainWindow::onStereoSettings()
{
    StereoSettingsDialog dlg(stereoConfig_, sgbmParams_, this);
    if (dlg.exec() == QDialog::Accepted) {
        stereoConfig_ = dlg.sourceConfig();
        sgbmParams_ = dlg.sgbmParams();
        thread_.setSGBMParams(sgbmParams_);

        if (thread_.isStereoMode()) {
            thread_.setStereoMode(false);
            if (thread_.openStereo(stereoConfig_)) {
                thread_.setStereoMode(true);
            } else {
                QMessageBox::warning(this, Lang::s("error"), Lang::s("stereo_open_fail"));
                stereoBtn_->setChecked(false);
            }
        }
    }
}

void MainWindow::onCalibrate()
{
    if (!thread_.isStereoMode()) return;

    CalibrationDialog dlg(&thread_.stereoSource(), this);
    connect(&dlg, &CalibrationDialog::capturedCountChanged, this, [this]() {
        // Could update UI during calibration
    });
    if (dlg.exec() == QDialog::Accepted) {
        StereoCalibration cal = dlg.result();
        if (cal.valid) {
            thread_.setStereoCalibration(cal);

            QString defaultName = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + "_stereo.yml";
            QString path = QFileDialog::getSaveFileName(this, Lang::s("calib_save"),
                defaultName, Lang::s("calib_file_filter"), nullptr, QFileDialog::DontUseNativeDialog);
            if (!path.isEmpty()) {
                StereoRectifier rectifier;
                rectifier.setCalibration(cal);
                if (rectifier.saveCalibration(path.toStdString())) {
                    lastCalibPath_ = path;
                    statusBar()->showMessage(Lang::s("calib_saved") + path, 3000);
                }
            }
        }
    }
}

void MainWindow::onToggleDepthOverlay(bool checked)
{
    thread_.setDepthOverlay(checked);
    depthOverlayBtn_->setText(checked ? Lang::s("depth_overlay_on") : Lang::s("depth_overlay_off"));
    if (checked) {
        depthDock_->show();
    }
}

void MainWindow::onDepthMapReady(const QImage& depthViz, float avgDepth)
{
    Q_UNUSED(avgDepth);

    // Update point cloud dock with depth visualization
    if (pointCloudDock_->isVisible() && !depthViz.isNull()) {
        pointCloudLabel_->setPixmap(QPixmap::fromImage(depthViz).scaled(
            pointCloudLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    // Update depth table from latest detections with distance
    if (depthDock_->isVisible()) {
        auto dets = thread_.lastDetections();

        // Filter detections with valid distance
        int row = 0;
        for (const auto& d : dets) {
            if (d.distance > 0) {
                depthTable_->setRowCount(row + 1);
                depthTable_->setItem(row, 0,
                    new QTableWidgetItem(QString::number(row)));
                depthTable_->setItem(row, 1,
                    new QTableWidgetItem(QString::fromStdString(YOLODetector::CLASS_NAMES[d.classId])));
                depthTable_->setItem(row, 2,
                    new QTableWidgetItem(QString::number(d.distance, 'f', 2)));
                depthTable_->setItem(row, 3,
                    new QTableWidgetItem(QString::number(d.confidence, 'f', 2)));
                row++;
            }
        }
        if (row == 0) depthTable_->setRowCount(0);
    }
}

void MainWindow::onPoseDataUpdated(const std::vector<Detection>& dets)
{
    if (!poseDock_->isVisible()) return;

    if (dets.empty()) {
        poseTable_->setRowCount(0);
        posePersonLabel_->setText("");
        return;
    }

    const auto& det = *std::max_element(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) { return a.confidence < b.confidence; });

    posePersonLabel_->setText(
        QString("Person conf: %1  |  Keypoints: %2")
            .arg(det.confidence, 0, 'f', 2)
            .arg(det.keypoints.size()));

    poseTable_->setRowCount((int)det.keypoints.size());
    for (int i = 0; i < (int)det.keypoints.size(); i++) {
        const auto& kp = det.keypoints[i];
        QString name = (i < (int)YOLODetector::KEYPOINT_NAMES.size())
            ? QString::fromStdString(YOLODetector::KEYPOINT_NAMES[i])
            : QString("kp_%1").arg(i);
        poseTable_->setItem(i, 0, new QTableWidgetItem(name));
        poseTable_->setItem(i, 1,
            new QTableWidgetItem(QString("(%1, %2)").arg((int)kp.pt.x).arg((int)kp.pt.y)));
        poseTable_->setItem(i, 2,
            new QTableWidgetItem(QString::number(kp.confidence, 'f', 2)));
    }
}

void MainWindow::onTrackingStatsUpdated(const QMap<int,int>& uniqueCounts, int totalUnique)
{
    uniqueCounts_ = uniqueCounts;
    totalUniqueLabel_->setText(Lang::s("stats_total_unique").arg(totalUnique));
}

void MainWindow::onDrawCountingLine()
{
    drawMode_ = DrawMode::WaitingPt1;
    videoLabel_->setCursor(Qt::CrossCursor);
    statusBar()->showMessage(Lang::s("click_pt1"), 10000);
}

void MainWindow::onClearCountingLine()
{
    thread_.clearCountingLine();
    thread_.resetCrossingCounts();
    countingTable_->setRowCount(0);
    statusBar()->showMessage(Lang::s("line_cleared"), 3000);
}

void MainWindow::onCrossingStatsUpdated(const QMap<int, QMap<int, int>>& counts)
{
    if (!countingDock_->isVisible()) return;

    countingTable_->setRowCount(counts.size());
    int row = 0;
    for (auto it = counts.constBegin(); it != counts.constEnd(); ++it) {
        int fwd = it.value().value(1, 0);
        int rev = it.value().value(-1, 0);
        countingTable_->setItem(row, 0,
            new QTableWidgetItem(QString::fromStdString(YOLODetector::CLASS_NAMES[it.key()])));
        countingTable_->setItem(row, 1, new QTableWidgetItem(QString::number(fwd)));
        countingTable_->setItem(row, 2, new QTableWidgetItem(QString::number(rev)));
        countingTable_->setItem(row, 3, new QTableWidgetItem(QString::number(fwd + rev)));
        row++;
    }
}

void MainWindow::onToggleLoop(bool checked)
{
    thread_.setLoopEnabled(checked);
    statusBar()->showMessage(
        checked ? Lang::s("loop_enabled") : Lang::s("loop_disabled"), 2000);
}

void MainWindow::onExport()
{
    // Show choice dialog when tracking is enabled
    bool exportTracking = false;
    if (thread_.isTrackingEnabled()) {
        QMessageBox choiceDlg(QMessageBox::Question, Lang::s("export_choice"),
            Lang::s("export_choice"), QMessageBox::NoButton, this);
        QPushButton* detBtn = choiceDlg.addButton(Lang::s("export_detect"), QMessageBox::AcceptRole);
        QPushButton* trackBtn = choiceDlg.addButton(Lang::s("export_tracking"), QMessageBox::AcceptRole);
        QPushButton* cancelBtn = choiceDlg.addButton(QMessageBox::Cancel);
        choiceDlg.exec();
        if (choiceDlg.clickedButton() == cancelBtn) return;
        exportTracking = (choiceDlg.clickedButton() == trackBtn);
    }

    if (exportTracking) {
        // Export tracking data
        auto history = thread_.trackHistory();
        auto uniqueCounts = thread_.isTrackingEnabled() ? uniqueCounts_ : QMap<int,int>();

        if (history.empty()) {
            QMessageBox::information(this, Lang::s("export_tracks"), Lang::s("export_track_no_data"));
            return;
        }

        QString defaultName = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + "_tracks";
        QString path = QFileDialog::getSaveFileName(this, Lang::s("export_tracks"),
            defaultName, Lang::s("export_track_filter"), nullptr, QFileDialog::DontUseNativeDialog);
        if (path.isEmpty()) return;

        bool success = false;
        if (path.endsWith(".csv", Qt::CaseInsensitive)) {
            QFile file(path);
            if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
                QTextStream out(&file);
                out << "trackId,class,classId,timestamp,x,y,width,height,speed,angle\n";
                for (const auto& r : history) {
                    QDateTime dt = QDateTime::fromMSecsSinceEpoch(r.timestampMs);
                    out << r.trackId << ","
                        << QString::fromStdString(YOLODetector::CLASS_NAMES[r.classId]) << ","
                        << r.classId << ","
                        << dt.toString("yyyy-MM-ddTHH:mm:ss.zzz") << ","
                        << r.x << "," << r.y << ","
                        << r.width << "," << r.height << ","
                        << QString::number(r.speed, 'f', 2) << ","
                        << QString::number(r.angle, 'f', 1) << "\n";
                }
                success = true;
            }
        } else {
            // Build JSON with tracks grouped by trackId
            QJsonObject root;
            root["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);

            // Group records by trackId
            QMap<int, QJsonArray> trackPoints;
            for (const auto& r : history) {
                QJsonObject pt;
                pt["t"] = QDateTime::fromMSecsSinceEpoch(r.timestampMs).toString(Qt::ISODate);
                pt["x"] = r.x;
                pt["y"] = r.y;
                pt["w"] = r.width;
                pt["h"] = r.height;
                pt["speed"] = qRound(r.speed * 100) / 100.0;
                pt["angle"] = qRound(r.angle * 10) / 10.0;
                trackPoints[r.trackId].append(pt);
            }

            // Build tracks array with class info from first record
            QMap<int, int> trackClassMap;
            for (const auto& r : history) {
                if (!trackClassMap.contains(r.trackId))
                    trackClassMap[r.trackId] = r.classId;
            }

            QJsonArray tracksArr;
            for (auto it = trackPoints.constBegin(); it != trackPoints.constEnd(); ++it) {
                QJsonObject tObj;
                tObj["trackId"] = it.key();
                int cid = trackClassMap.value(it.key(), 0);
                tObj["class"] = QString::fromStdString(YOLODetector::CLASS_NAMES[cid]);
                tObj["classId"] = cid;
                tObj["points"] = it.value();
                tracksArr.append(tObj);
            }
            root["tracks"] = tracksArr;

            // Unique counts
            QJsonObject ucObj;
            for (auto it = uniqueCounts.constBegin(); it != uniqueCounts.constEnd(); ++it)
                ucObj[QString::fromStdString(YOLODetector::CLASS_NAMES[it.key()])] = it.value();
            root["uniqueCounts"] = ucObj;

            QFile file(path);
            if (file.open(QIODevice::WriteOnly)) {
                file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
                success = true;
            }
        }

        if (success)
            statusBar()->showMessage(Lang::s("export_track_done") + path, 3000);
        else
            QMessageBox::warning(this, Lang::s("error"), Lang::s("export_fail"));
    } else {
        // Export detections (original behavior)
        auto dets = thread_.lastDetections();
        if (dets.empty()) {
            QMessageBox::information(this, Lang::s("export_title"), Lang::s("export_no_data"));
            return;
        }

        QString path = QFileDialog::getSaveFileName(this, Lang::s("export_title"),
            QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"),
            Lang::s("export_filter"), nullptr, QFileDialog::DontUseNativeDialog);
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
                if (!d.keypoints.empty()) {
                    QJsonArray kpArr;
                    for (int k = 0; k < (int)d.keypoints.size(); k++) {
                        QJsonObject kpObj;
                        QString kpName = (k < (int)YOLODetector::KEYPOINT_NAMES.size())
                            ? QString::fromStdString(YOLODetector::KEYPOINT_NAMES[k])
                            : QString("kp_%1").arg(k);
                        kpObj["name"] = kpName;
                        kpObj["x"] = qRound(d.keypoints[k].pt.x * 10) / 10.0;
                        kpObj["y"] = qRound(d.keypoints[k].pt.y * 10) / 10.0;
                        kpObj["confidence"] = qRound(d.keypoints[k].confidence * 1000) / 1000.0;
                        kpArr.append(kpObj);
                    }
                    obj["keypoints"] = kpArr;
                }
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
    if (thread_.detector().isPoseModel()) {
        QMessageBox::about(this, Lang::s("about"),
            Lang::s("about_text_pose").arg(qVersion()).arg(cvVer));
    } else {
        QMessageBox::about(this, Lang::s("about"),
            Lang::s("about_text").arg(qVersion()).arg(cvVer));
    }
}

void MainWindow::updateModelTypeUI()
{
    bool isPose = thread_.detector().isPoseModel();
    bool isStereo = thread_.isStereoMode();

    poseDock_->setVisible(isPose && !isStereo);
    skeletonBtn_->setVisible(isPose);
    classFilterBtn_->setVisible(!isPose);

    calibrateBtn_->setVisible(isStereo);
    depthOverlayBtn_->setVisible(isStereo);
    stereoSettingsBtn_->setVisible(isStereo);

    if (isStereo) {
        depthDock_->setVisible(true);
        pointCloudDock_->setVisible(true);
    }

    if (isPose)
        statusBar()->showMessage(Lang::s("pose_model_loaded"), 3000);
    else
        statusBar()->showMessage(Lang::s("detection_model_loaded"), 3000);

    QString devText = thread_.detector().isGpuEnabled()
        ? Lang::s("device_gpu") : Lang::s("device_cpu");
    if (isPose) devText += " | Pose";
    if (isStereo) devText += " | Stereo";
    deviceLabel_->setText(devText);
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
    trajectoryBtn_->setText(trajectoryBtn_->isChecked() ? Lang::s("trajectory_on") : Lang::s("trajectory_off"));
    speedBtn_->setText(speedBtn_->isChecked() ? Lang::s("speed_on") : Lang::s("speed_off"));
    skeletonBtn_->setText(skeletonBtn_->isChecked() ? Lang::s("skeleton_on") : Lang::s("skeleton_off"));
    stereoBtn_->setText(stereoBtn_->isChecked() ? Lang::s("stereo_on") : Lang::s("stereo_off"));
    calibrateBtn_->setText(Lang::s("calibration"));
    depthOverlayBtn_->setText(depthOverlayBtn_->isChecked() ? Lang::s("depth_overlay_on") : Lang::s("depth_overlay_off"));
    stereoSettingsBtn_->setText(Lang::s("stereo_settings"));
    countLineBtn_->setText(Lang::s("draw_line"));
    clearLineBtn_->setText(Lang::s("clear_line"));
    langBtn_->setText(Lang::s("lang_toggle"));
    clearStatsBtn_->setText(Lang::s("stats_clear"));
    resetCountsBtn_->setText(Lang::s("reset_counts"));
    clearCrossingBtn_->setText(Lang::s("clear_line"));
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
    trajectoryBtn_->setToolTip(Lang::s("tip_trajectory"));
    speedBtn_->setToolTip(Lang::s("tip_speed"));
    skeletonBtn_->setToolTip(Lang::s("tip_skeleton"));
    stereoBtn_->setToolTip(Lang::s("tip_stereo"));
    calibrateBtn_->setToolTip(Lang::s("tip_calibrate"));
    depthOverlayBtn_->setToolTip(Lang::s("tip_depth_overlay"));
    countLineBtn_->setToolTip(Lang::s("tip_draw_line"));
    clearLineBtn_->setToolTip(Lang::s("tip_clear_line"));
    langBtn_->setToolTip(Lang::s("tip_lang"));

    statsDock_->setWindowTitle(Lang::s("stats_title"));
    statsTable_->setHorizontalHeaderLabels(
        {Lang::s("stats_class"), Lang::s("stats_count"), Lang::s("stats_unique")});
    totalUniqueLabel_->setText(Lang::s("stats_total_unique").arg(
        [] (const QMap<int,int>& m) { int t=0; for (auto v : m) t += v; return t; } (uniqueCounts_)));

    countingDock_->setWindowTitle(Lang::s("crossing_count"));
    countingTable_->setHorizontalHeaderLabels(
        {Lang::s("stats_class"), Lang::s("forward"), Lang::s("reverse"), Lang::s("total")});

    poseDock_->setWindowTitle(Lang::s("pose_title"));
    poseTable_->setHorizontalHeaderLabels(
        {Lang::s("pose_keypoint"), Lang::s("pose_position"), Lang::s("pose_confidence")});

    depthDock_->setWindowTitle(Lang::s("depth_dock"));
    depthTable_->setHorizontalHeaderLabels(
        {Lang::s("depth_track_id"), Lang::s("depth_class"),
         Lang::s("depth_dist"), Lang::s("depth_conf")});
    pointCloudDock_->setWindowTitle(Lang::s("point_cloud_dock"));

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
