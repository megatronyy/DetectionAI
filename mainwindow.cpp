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
#include <QFileInfo>
#include <QHeaderView>
#include <QMenuBar>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <algorithm>
#include <QMimeData>
#include <QUrl>
#include <QProgressDialog>
#include <opencv2/core/version.hpp>

// --- createAction helper ---

QAction* MainWindow::createAction(const QString& text, const QString& tip,
                                   const QKeySequence& shortcut,
                                   bool checkable, bool checked)
{
    QAction* a = new QAction(text, this);
    a->setToolTip(tip);
    if (!shortcut.isEmpty()) {
        a->setShortcut(shortcut);
        a->setShortcutContext(Qt::ApplicationShortcut);
    }
    a->setCheckable(checkable);
    if (checkable) a->setChecked(checked);
    return a;
}

// --- Constructor / Destructor ---

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

// --- enumerateCameras ---

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

// --- setupUI (rewritten) ---

void MainWindow::setupUI()
{
    // === Create all QActions ===

    actPause_       = createAction(Lang::s("pause"), Lang::s("tip_pause"), Qt::Key_Space);
    actScreenshot_  = createAction(Lang::s("screenshot"), Lang::s("tip_screenshot"), Qt::Key_S);
    actRecord_      = createAction(Lang::s("record"), Lang::s("tip_record"), Qt::Key_R);
    actExport_      = createAction(Lang::s("export_btn"), Lang::s("tip_export"), QKeySequence("Ctrl+E"));
    actOpenVideo_   = createAction(Lang::s("open_video"), Lang::s("tip_open_video"), QKeySequence("Ctrl+O"));
    actNetworkCam_  = createAction(Lang::s("network_cam"), Lang::s("tip_network"), QKeySequence("Ctrl+N"));
    actLoop_        = createAction(Lang::s("loop"), Lang::s("tip_loop"), Qt::Key_L, true);
    actSwitchModel_ = createAction(Lang::s("switch_model"), Lang::s("tip_model"), QKeySequence("Ctrl+M"));
    actClassFilter_ = createAction(Lang::s("class_filter"), Lang::s("tip_filter"));
    actTracking_    = createAction(Lang::s("tracking_off"), Lang::s("tip_tracking"), Qt::Key_T, true);
    actTrajectory_  = createAction(Lang::s("trajectory_off"), Lang::s("tip_trajectory"), QKeySequence("Shift+T"), true);
    actSpeed_       = createAction(Lang::s("speed_off"), Lang::s("tip_speed"), QKeySequence("Shift+S"), true);
    actSkeleton_    = createAction(Lang::s("skeleton_off"), Lang::s("tip_skeleton"), Qt::Key_K, true, true);
    actStereo_      = createAction(Lang::s("stereo_off"), Lang::s("tip_stereo"), Qt::Key_B, true);
    actDepthOverlay_= createAction(Lang::s("depth_overlay_off"), Lang::s("tip_depth_overlay"), Qt::Key_D, true);
    actCalibrate_   = createAction(Lang::s("calibration"), Lang::s("tip_calibrate"), QKeySequence("Shift+B"));
    actStereoSettings_ = createAction(Lang::s("stereo_settings"), Lang::s("stereo_settings"));
    actCountLine_   = createAction(Lang::s("draw_line"), Lang::s("tip_draw_line"), Qt::Key_C);
    actClearLine_   = createAction(Lang::s("clear_line"), Lang::s("tip_clear_line"), QKeySequence("Shift+C"));
    actLanguage_    = createAction(Lang::s("lang_toggle"), Lang::s("tip_lang"));
    actFullScreen_  = createAction(Lang::s("menu_fullscreen"), QString(), Qt::Key_F11, true);
    actExit_        = createAction(Lang::s("menu_exit"), QString(), QKeySequence("Ctrl+Q"));

    actSkeleton_->setVisible(false);
    actCalibrate_->setVisible(false);
    actDepthOverlay_->setVisible(false);
    actStereoSettings_->setVisible(false);

    // === Connect actions to slots ===

    connect(actPause_, &QAction::triggered, this, &MainWindow::onTogglePause);
    connect(actScreenshot_, &QAction::triggered, this, &MainWindow::onScreenshot);
    connect(actRecord_, &QAction::triggered, this, &MainWindow::onToggleRecord);
    connect(actExport_, &QAction::triggered, this, &MainWindow::onExport);
    connect(actOpenVideo_, &QAction::triggered, this, &MainWindow::onOpenVideo);
    connect(actNetworkCam_, &QAction::triggered, this, &MainWindow::onNetworkCamera);
    connect(actLoop_, &QAction::toggled, this, &MainWindow::onToggleLoop);
    connect(actSwitchModel_, &QAction::triggered, this, &MainWindow::onSwitchModel);
    connect(actClassFilter_, &QAction::triggered, this, &MainWindow::onClassFilter);
    connect(actTracking_, &QAction::toggled, this, &MainWindow::onToggleTracking);
    connect(actTrajectory_, &QAction::toggled, this, &MainWindow::onToggleTrajectory);
    connect(actSpeed_, &QAction::toggled, this, &MainWindow::onToggleSpeed);
    connect(actSkeleton_, &QAction::toggled, this, &MainWindow::onToggleSkeleton);
    connect(actStereo_, &QAction::toggled, this, &MainWindow::onToggleStereo);
    connect(actDepthOverlay_, &QAction::toggled, this, &MainWindow::onToggleDepthOverlay);
    connect(actCalibrate_, &QAction::triggered, this, &MainWindow::onCalibrate);
    connect(actStereoSettings_, &QAction::triggered, this, &MainWindow::onStereoSettings);
    connect(actCountLine_, &QAction::triggered, this, &MainWindow::onDrawCountingLine);
    connect(actClearLine_, &QAction::triggered, this, &MainWindow::onClearCountingLine);
    connect(actLanguage_, &QAction::triggered, this, &MainWindow::onToggleLanguage);
    connect(actFullScreen_, &QAction::toggled, this, &MainWindow::onToggleFullScreen);
    connect(actExit_, &QAction::triggered, this, &QWidget::close);

    // === Menu bar ===

    fileMenu_ = menuBar()->addMenu(Lang::s("menu_file"));
    fileMenu_->addAction(actOpenVideo_);
    fileMenu_->addAction(actNetworkCam_);
    fileMenu_->addSeparator();
    cameraCombo_ = new QComboBox;
    enumerateCameras();
    cameraCombo_->setCurrentIndex(0);
    connect(cameraCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onCameraChanged);
    fileMenu_->addSeparator();
    fileMenu_->addAction(actExport_);
    fileMenu_->addSeparator();
    fileMenu_->addAction(actExit_);

    modelMenu_ = menuBar()->addMenu(Lang::s("menu_model"));
    modelMenu_->addAction(actSwitchModel_);
    recentModelsMenu_ = modelMenu_->addMenu(Lang::s("recent_models"));
    modelMenu_->addSeparator();
    modelMenu_->addAction(actClassFilter_);

    playbackMenu_ = menuBar()->addMenu(Lang::s("menu_playback"));
    playbackMenu_->addAction(actPause_);
    playbackMenu_->addAction(actScreenshot_);
    playbackMenu_->addAction(actRecord_);
    playbackMenu_->addAction(actLoop_);

    trackingMenu_ = menuBar()->addMenu(Lang::s("menu_tracking"));
    trackingMenu_->addAction(actTracking_);
    trackingMenu_->addAction(actTrajectory_);
    trackingMenu_->addAction(actSpeed_);
    trackingMenu_->addSeparator();
    trackingMenu_->addAction(actCountLine_);
    trackingMenu_->addAction(actClearLine_);

    stereoMenu_ = menuBar()->addMenu(Lang::s("menu_stereo"));
    stereoMenu_->addAction(actStereo_);
    stereoMenu_->addAction(actDepthOverlay_);
    stereoMenu_->addSeparator();
    stereoMenu_->addAction(actCalibrate_);
    stereoMenu_->addAction(actStereoSettings_);

    viewMenu_ = menuBar()->addMenu(Lang::s("menu_view"));
    viewMenu_->addAction(actFullScreen_);
    viewMenu_->addSeparator();
    viewMenu_->addAction(actLanguage_);

    helpMenu_ = menuBar()->addMenu(Lang::s("menu_help"));
    helpMenu_->addAction(Lang::s("about"), this, &MainWindow::onAbout);

    // === Toolbar (compact: 6 actions + camera combo) ===

    QToolBar* toolbar = addToolBar("Controls");
    toolbar->setMovable(false);
    toolbar->addAction(actPause_);
    toolbar->addSeparator();
    toolbar->addAction(actScreenshot_);
    toolbar->addAction(actRecord_);
    toolbar->addAction(actExport_);
    toolbar->addSeparator();
    toolbar->addWidget(new QLabel(Lang::s("input_source")));
    toolbar->addWidget(cameraCombo_);
    toolbar->addSeparator();
    toolbar->addAction(actLanguage_);

    // === Sliders ===

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

    // === Video display ===

    videoLabel_ = new QLabel;
    videoLabel_->setAlignment(Qt::AlignCenter);
    videoLabel_->setStyleSheet("QLabel { background-color: #1a1a1a; }");
    videoLabel_->installEventFilter(this);

    // === Central layout ===

    QWidget* central = new QWidget;
    QVBoxLayout* mainLayout = new QVBoxLayout(central);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);
    mainLayout->addWidget(sliderWidget);
    mainLayout->addWidget(videoLabel_, 1);
    setCentralWidget(central);

    // === Status bar ===

    fpsLabel_    = new QLabel("FPS: --");
    detLabel_    = new QLabel(Lang::s("det_count").arg(0));
    inferLabel_  = new QLabel(Lang::s("infer_ms").arg(0));
    deviceLabel_ = new QLabel(Lang::s("device_cpu"));

    statusBar()->addWidget(fpsLabel_);
    statusBar()->addWidget(detLabel_);
    statusBar()->addWidget(inferLabel_);
    statusBar()->addPermanentWidget(deviceLabel_);

    // === Unified right panel (QDockWidget + QTabWidget) ===

    panelDock_ = new QDockWidget(Lang::s("app_title"), this);
    panelTabs_ = new QTabWidget(panelDock_);

    // --- Tab 1: Statistics ---
    statsPage_ = new QWidget;
    auto* statsLayout = new QVBoxLayout(statsPage_);
    statsLayout->setContentsMargins(0, 0, 0, 0);

    statsTable_ = new QTableWidget(0, 3, statsPage_);
    statsTable_->setHorizontalHeaderLabels(
        {Lang::s("stats_class"), Lang::s("stats_count"), Lang::s("stats_unique")});
    statsTable_->horizontalHeader()->setStretchLastSection(true);
    statsTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    statsTable_->setSelectionBehavior(QAbstractItemView::SelectRows);

    totalUniqueLabel_ = new QLabel(Lang::s("stats_total_unique").arg(0));

    auto* statsBtnLayout = new QHBoxLayout;
    auto* clearStatsBtn = new QPushButton(Lang::s("stats_clear"));
    auto* resetCountsBtn = new QPushButton(Lang::s("reset_counts"));
    statsBtnLayout->addWidget(clearStatsBtn);
    statsBtnLayout->addWidget(resetCountsBtn);

    statsLayout->addWidget(statsTable_);
    statsLayout->addWidget(totalUniqueLabel_);
    statsLayout->addLayout(statsBtnLayout);

    connect(clearStatsBtn, &QPushButton::clicked, this, &MainWindow::onClearStats);
    connect(resetCountsBtn, &QPushButton::clicked, this, [this]() {
        thread_.resetTrackCounts();
        uniqueCounts_.clear();
        totalUniqueLabel_->setText(Lang::s("stats_total_unique").arg(0));
        for (int r = 0; r < statsTable_->rowCount(); r++)
            statsTable_->setItem(r, 2, new QTableWidgetItem("0"));
    });

    // --- Tab 2: Tracking + Counting ---
    trackingPage_ = new QWidget;
    auto* trackLayout = new QVBoxLayout(trackingPage_);
    trackLayout->setContentsMargins(0, 0, 0, 0);

    auto* trackBtnRow = new QHBoxLayout;
    auto* toggleTrackBtn = new QPushButton(Lang::s("tracking_off"));
    toggleTrackBtn->setCheckable(true);
    auto* toggleTrajectoryBtn = new QPushButton(Lang::s("trajectory_off"));
    toggleTrajectoryBtn->setCheckable(true);
    auto* toggleSpeedBtn = new QPushButton(Lang::s("speed_off"));
    toggleSpeedBtn->setCheckable(true);
    trackBtnRow->addWidget(toggleTrackBtn);
    trackBtnRow->addWidget(toggleTrajectoryBtn);
    trackBtnRow->addWidget(toggleSpeedBtn);
    trackBtnRow->addStretch();
    trackLayout->addLayout(trackBtnRow);

    // Sync tab toggle buttons with menu actions
    connect(toggleTrackBtn, &QPushButton::toggled, this, [this](bool checked) {
        actTracking_->setChecked(checked);
    });
    connect(actTracking_, &QAction::toggled, this, [toggleTrackBtn](bool checked) {
        toggleTrackBtn->blockSignals(true);
        toggleTrackBtn->setChecked(checked);
        toggleTrackBtn->setText(checked ? Lang::s("tracking_on") : Lang::s("tracking_off"));
        toggleTrackBtn->blockSignals(false);
    });
    connect(toggleTrajectoryBtn, &QPushButton::toggled, this, [this](bool checked) {
        actTrajectory_->setChecked(checked);
    });
    connect(actTrajectory_, &QAction::toggled, this, [toggleTrajectoryBtn](bool checked) {
        toggleTrajectoryBtn->blockSignals(true);
        toggleTrajectoryBtn->setChecked(checked);
        toggleTrajectoryBtn->setText(checked ? Lang::s("trajectory_on") : Lang::s("trajectory_off"));
        toggleTrajectoryBtn->blockSignals(false);
    });
    connect(toggleSpeedBtn, &QPushButton::toggled, this, [this](bool checked) {
        actSpeed_->setChecked(checked);
    });
    connect(actSpeed_, &QAction::toggled, this, [toggleSpeedBtn](bool checked) {
        toggleSpeedBtn->blockSignals(true);
        toggleSpeedBtn->setChecked(checked);
        toggleSpeedBtn->setText(checked ? Lang::s("speed_on") : Lang::s("speed_off"));
        toggleSpeedBtn->blockSignals(false);
    });

    countingTable_ = new QTableWidget(0, 4, trackingPage_);
    countingTable_->setHorizontalHeaderLabels(
        {Lang::s("stats_class"), Lang::s("forward"), Lang::s("reverse"), Lang::s("total")});
    countingTable_->horizontalHeader()->setStretchLastSection(true);
    countingTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    countingTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    trackLayout->addWidget(countingTable_);

    auto* countBtnLayout = new QHBoxLayout;
    auto* drawLineBtn = new QPushButton(Lang::s("draw_line"));
    auto* clearLineBtn = new QPushButton(Lang::s("clear_line"));
    countBtnLayout->addWidget(drawLineBtn);
    countBtnLayout->addWidget(clearLineBtn);
    trackLayout->addLayout(countBtnLayout);

    connect(drawLineBtn, &QPushButton::clicked, this, &MainWindow::onDrawCountingLine);
    connect(clearLineBtn, &QPushButton::clicked, this, [this]() {
        thread_.clearCountingLine();
        thread_.resetCrossingCounts();
        countingTable_->setRowCount(0);
    });

    // --- Tab 3: Pose ---
    posePage_ = new QWidget;
    auto* poseLayout = new QVBoxLayout(posePage_);
    poseLayout->setContentsMargins(0, 0, 0, 0);

    auto* poseBtnRow = new QHBoxLayout;
    auto* toggleSkeletonBtn = new QPushButton(Lang::s("skeleton_off"));
    toggleSkeletonBtn->setCheckable(true);
    toggleSkeletonBtn->setChecked(true);
    poseBtnRow->addWidget(toggleSkeletonBtn);
    poseBtnRow->addStretch();
    poseLayout->addLayout(poseBtnRow);

    connect(toggleSkeletonBtn, &QPushButton::toggled, this, [this](bool checked) {
        actSkeleton_->setChecked(checked);
    });
    connect(actSkeleton_, &QAction::toggled, this, [toggleSkeletonBtn](bool checked) {
        toggleSkeletonBtn->blockSignals(true);
        toggleSkeletonBtn->setChecked(checked);
        toggleSkeletonBtn->setText(checked ? Lang::s("skeleton_on") : Lang::s("skeleton_off"));
        toggleSkeletonBtn->blockSignals(false);
    });

    posePersonLabel_ = new QLabel("");
    poseLayout->addWidget(posePersonLabel_);

    poseTable_ = new QTableWidget(0, 3, posePage_);
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

    // --- Tab 4: Depth ---
    depthPage_ = new QWidget;
    auto* depthLayout = new QVBoxLayout(depthPage_);
    depthLayout->setContentsMargins(0, 0, 0, 0);

    auto* depthBtnRow = new QHBoxLayout;
    auto* toggleDepthBtn = new QPushButton(Lang::s("depth_overlay_off"));
    toggleDepthBtn->setCheckable(true);
    depthBtnRow->addWidget(toggleDepthBtn);
    depthBtnRow->addStretch();
    depthLayout->addLayout(depthBtnRow);

    connect(toggleDepthBtn, &QPushButton::toggled, this, [this](bool checked) {
        actDepthOverlay_->setChecked(checked);
    });
    connect(actDepthOverlay_, &QAction::toggled, this, [toggleDepthBtn](bool checked) {
        toggleDepthBtn->blockSignals(true);
        toggleDepthBtn->setChecked(checked);
        toggleDepthBtn->setText(checked ? Lang::s("depth_overlay_on") : Lang::s("depth_overlay_off"));
        toggleDepthBtn->blockSignals(false);
    });

    depthTable_ = new QTableWidget(0, 4, depthPage_);
    depthTable_->setHorizontalHeaderLabels(
        {Lang::s("depth_track_id"), Lang::s("depth_class"),
         Lang::s("depth_dist"), Lang::s("depth_conf")});
    depthTable_->horizontalHeader()->setStretchLastSection(true);
    depthTable_->setEditTriggers(QAbstractItemView::NoEditTriggers);
    depthTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    depthLayout->addWidget(depthTable_);

    pointCloudLabel_ = new QLabel;
    pointCloudLabel_->setAlignment(Qt::AlignCenter);
    pointCloudLabel_->setStyleSheet("QLabel { background-color: #1a1a1a; }");
    pointCloudLabel_->setMinimumSize(300, 200);
    depthLayout->addWidget(pointCloudLabel_);

    // --- Assemble tabs ---
    panelTabs_->addTab(statsPage_, Lang::s("stats_title"));
    panelTabs_->addTab(trackingPage_, Lang::s("crossing_count"));
    panelTabs_->addTab(posePage_, Lang::s("pose_title"));
    panelTabs_->addTab(depthPage_, Lang::s("depth_dock"));

    panelDock_->setWidget(panelTabs_);
    addDockWidget(Qt::RightDockWidgetArea, panelDock_);

    // Hide conditional tabs initially
    panelTabs_->setTabVisible(2, false); // pose
    panelTabs_->setTabVisible(3, false); // depth
}

// --- Settings ---

void MainWindow::loadSettings()
{
    QSettings settings("DetectionAI", "YOLODetector");
    float conf = settings.value("confidence", 0.25).toFloat();
    float iou = settings.value("iou", 0.45).toFloat();
    int cam = settings.value("camera", 0).toInt();
    QSize winSize = settings.value("windowSize", QSize(960, 720)).toSize();
    QPoint winPos = settings.value("windowPos", QPoint()).toPoint();
    int lang = settings.value("language", 0).toInt();

    Lang::setLanguage(static_cast<Lang::Language>(lang));

    confSlider_->setValue((int)(conf * 100));
    iouSlider_->setValue((int)(iou * 100));
    cameraCombo_->setCurrentIndex(std::min(cam, cameraCombo_->count() - 1));
    resize(winSize);
    if (!winPos.isNull()) move(winPos);

    actTracking_->setChecked(settings.value("tracking", false).toBool());
    thread_.setTrackingEnabled(actTracking_->isChecked());
    actTrajectory_->setChecked(settings.value("trajectory", false).toBool());
    thread_.setTrajectoryEnabled(actTrajectory_->isChecked());
    actSpeed_->setChecked(settings.value("speed", false).toBool());
    thread_.setSpeedEnabled(actSpeed_->isChecked());
    actSkeleton_->setChecked(settings.value("skeleton", true).toBool());
    thread_.setSkeletonEnabled(actSkeleton_->isChecked());
    actStereo_->setChecked(settings.value("stereoMode", false).toBool());
    thread_.setStereoMode(actStereo_->isChecked());
    actDepthOverlay_->setChecked(settings.value("depthOverlay", false).toBool());
    thread_.setDepthOverlay(actDepthOverlay_->isChecked());
    float kpConf = settings.value("keypointConfThreshold", 0.5).toFloat();
    kpConfSlider_->setValue((int)(kpConf * 100));
    thread_.setKeypointConfThreshold(kpConf);
    actLoop_->setChecked(settings.value("loop", false).toBool());
    thread_.setLoopEnabled(actLoop_->isChecked());

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

    // Panel visibility
    bool panelVisible = settings.value("panelVisible", true).toBool();
    if (!panelVisible) panelDock_->hide();
    int panelTab = settings.value("panelTab", 0).toInt();
    if (panelTab >= 0 && panelTab < panelTabs_->count())
        panelTabs_->setCurrentIndex(panelTab);

    actFullScreen_->setChecked(isFullScreen());
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
    settings.setValue("tracking", actTracking_->isChecked());
    settings.setValue("trajectory", actTrajectory_->isChecked());
    settings.setValue("speed", actSpeed_->isChecked());
    settings.setValue("skeleton", actSkeleton_->isChecked());
    settings.setValue("keypointConfThreshold", thread_.keypointConfThreshold());
    settings.setValue("loop", actLoop_->isChecked());
    settings.setValue("language", static_cast<int>(Lang::language()));
    settings.setValue("stereoMode", actStereo_->isChecked());
    settings.setValue("depthOverlay", actDepthOverlay_->isChecked());
    settings.setValue("panelVisible", panelDock_->isVisible());
    settings.setValue("panelTab", panelTabs_->currentIndex());
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

// --- Events ---

void MainWindow::closeEvent(QCloseEvent* event)
{
    saveSettings();
    thread_.stop();
    thread_.wait();
    event->accept();
}

void MainWindow::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Escape) {
        if (drawMode_ != DrawMode::Idle) {
            drawMode_ = DrawMode::Idle;
            videoLabel_->setCursor(Qt::ArrowCursor);
            statusBar()->showMessage(Lang::s("draw_cancelled"), 3000);
        } else if (isFullScreen()) {
            showNormal();
            actFullScreen_->setChecked(false);
        } else {
            close();
        }
    } else {
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

    if (panelDock_->isVisible() && panelTabs_->currentIndex() == 0) {
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
    actPause_->setText(paused_ ? Lang::s("resume") : Lang::s("pause"));
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
    actPause_->setText(Lang::s("pause"));
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
    actPause_->setText(Lang::s("pause"));
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
        actRecord_->setText(Lang::s("record"));
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
            actRecord_->setText(Lang::s("stop_record"));
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
    actPause_->setText(Lang::s("pause"));
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
    actPause_->setText(Lang::s("pause"));
    thread_.start();
    statusBar()->showMessage(Lang::s("model_switched") + path, 3000);
}

void MainWindow::onRecentModel()
{
    // Delegate to the menu action
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
    actTracking_->setText(checked ? Lang::s("tracking_on") : Lang::s("tracking_off"));
    statusBar()->showMessage(
        checked ? Lang::s("tracking_enabled") : Lang::s("tracking_disabled"), 2000);
}

void MainWindow::onToggleTrajectory(bool checked)
{
    thread_.setTrajectoryEnabled(checked);
    actTrajectory_->setText(checked ? Lang::s("trajectory_on") : Lang::s("trajectory_off"));
    statusBar()->showMessage(
        checked ? Lang::s("trajectory_on") : Lang::s("trajectory_off"), 2000);
}

void MainWindow::onToggleSpeed(bool checked)
{
    thread_.setSpeedEnabled(checked);
    actSpeed_->setText(checked ? Lang::s("speed_on") : Lang::s("speed_off"));
    statusBar()->showMessage(
        checked ? Lang::s("speed_on") : Lang::s("speed_off"), 2000);
}

void MainWindow::onToggleSkeleton(bool checked)
{
    thread_.setSkeletonEnabled(checked);
    actSkeleton_->setText(checked ? Lang::s("skeleton_on") : Lang::s("skeleton_off"));
}

void MainWindow::onToggleStereo(bool checked)
{
    if (checked) {
        // Show settings dialog first if hardware not configured
        if (stereoConfig_.hardware == StereoHardware::SingleMono) {
            StereoSettingsDialog dlg(stereoConfig_, sgbmParams_, this);
            if (dlg.exec() != QDialog::Accepted) {
                actStereo_->setChecked(false);
                return;
            }
            stereoConfig_ = dlg.sourceConfig();
            sgbmParams_ = dlg.sgbmParams();
            thread_.setSGBMParams(sgbmParams_);
        }

        if (!thread_.openStereo(stereoConfig_)) {
            QMessageBox::warning(this, Lang::s("error"), Lang::s("stereo_open_fail"));
            actStereo_->setChecked(false);
            return;
        }
        thread_.setStereoMode(true);
        statusBar()->showMessage(Lang::s("stereo_connected"), 3000);
    } else {
        thread_.setStereoMode(false);
        thread_.openCamera(cameraCombo_->currentData().toInt());
        statusBar()->showMessage(Lang::s("stereo_disconnected"), 3000);
    }

    actStereo_->setText(checked ? Lang::s("stereo_on") : Lang::s("stereo_off"));
    actCalibrate_->setVisible(checked);
    actDepthOverlay_->setVisible(checked);
    actStereoSettings_->setVisible(checked);
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
                actStereo_->setChecked(false);
            }
        }
    }
}

void MainWindow::onCalibrate()
{
    if (!thread_.isStereoMode()) return;

    CalibrationDialog dlg(&thread_.stereoSource(), this);
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
    actDepthOverlay_->setText(checked ? Lang::s("depth_overlay_on") : Lang::s("depth_overlay_off"));
}

void MainWindow::onDepthMapReady(const QImage& depthViz, float avgDepth)
{
    Q_UNUSED(avgDepth);

    if (panelDock_->isVisible() && panelTabs_->currentIndex() == 3 && !depthViz.isNull()) {
        pointCloudLabel_->setPixmap(QPixmap::fromImage(depthViz).scaled(
            pointCloudLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    if (panelDock_->isVisible() && panelTabs_->currentIndex() == 3) {
        auto dets = thread_.lastDetections();
        int row = 0;
        for (const auto& d : dets) {
            if (d.distance > 0) {
                depthTable_->setRowCount(row + 1);
                depthTable_->setItem(row, 0, new QTableWidgetItem(QString::number(row)));
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
    if (!panelDock_->isVisible() || panelTabs_->currentIndex() != 2) return;

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
    if (!panelDock_->isVisible() || panelTabs_->currentIndex() != 1) return;

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
    bool exportTracking = false;
    if (thread_.isTrackingEnabled()) {
        QMessageBox choiceDlg(QMessageBox::Question, Lang::s("export_choice"),
            Lang::s("export_choice"), QMessageBox::NoButton, this);
        choiceDlg.addButton(Lang::s("export_detect"), QMessageBox::AcceptRole);
        QPushButton* trackBtn = choiceDlg.addButton(Lang::s("export_tracking"), QMessageBox::AcceptRole);
        QPushButton* cancelBtn = choiceDlg.addButton(QMessageBox::Cancel);
        choiceDlg.exec();
        if (choiceDlg.clickedButton() == cancelBtn) return;
        exportTracking = (choiceDlg.clickedButton() == trackBtn);
    }

    if (exportTracking) {
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
            QJsonObject root;
            root["timestamp"] = QDateTime::currentDateTime().toString(Qt::ISODate);

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

void MainWindow::onToggleFullScreen()
{
    if (actFullScreen_->isChecked())
        showFullScreen();
    else
        showNormal();
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

    actSkeleton_->setVisible(isPose);
    actClassFilter_->setVisible(!isPose);
    actCalibrate_->setVisible(isStereo);
    actDepthOverlay_->setVisible(isStereo);
    actStereoSettings_->setVisible(isStereo);

    panelTabs_->setTabVisible(2, isPose);
    panelTabs_->setTabVisible(3, isStereo);

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

    // Menu titles
    fileMenu_->setTitle(Lang::s("menu_file"));
    modelMenu_->setTitle(Lang::s("menu_model"));
    playbackMenu_->setTitle(Lang::s("menu_playback"));
    trackingMenu_->setTitle(Lang::s("menu_tracking"));
    stereoMenu_->setTitle(Lang::s("menu_stereo"));
    viewMenu_->setTitle(Lang::s("menu_view"));
    helpMenu_->setTitle(Lang::s("menu_help"));

    // Action texts
    actPause_->setText(paused_ ? Lang::s("resume") : Lang::s("pause"));
    actPause_->setToolTip(Lang::s("tip_pause"));
    actScreenshot_->setText(Lang::s("screenshot"));
    actScreenshot_->setToolTip(Lang::s("tip_screenshot"));
    actRecord_->setText(thread_.isRecording() ? Lang::s("stop_record") : Lang::s("record"));
    actRecord_->setToolTip(Lang::s("tip_record"));
    actExport_->setText(Lang::s("export_btn"));
    actExport_->setToolTip(Lang::s("tip_export"));
    actOpenVideo_->setText(Lang::s("open_video"));
    actOpenVideo_->setToolTip(Lang::s("tip_open_video"));
    actNetworkCam_->setText(Lang::s("network_cam"));
    actNetworkCam_->setToolTip(Lang::s("tip_network"));
    actLoop_->setText(Lang::s("loop"));
    actLoop_->setToolTip(Lang::s("tip_loop"));
    actSwitchModel_->setText(Lang::s("switch_model"));
    actSwitchModel_->setToolTip(Lang::s("tip_model"));
    recentModelsMenu_->setTitle(Lang::s("recent_models"));
    actClassFilter_->setText(Lang::s("class_filter"));
    actClassFilter_->setToolTip(Lang::s("tip_filter"));
    actTracking_->setText(actTracking_->isChecked() ? Lang::s("tracking_on") : Lang::s("tracking_off"));
    actTracking_->setToolTip(Lang::s("tip_tracking"));
    actTrajectory_->setText(actTrajectory_->isChecked() ? Lang::s("trajectory_on") : Lang::s("trajectory_off"));
    actTrajectory_->setToolTip(Lang::s("tip_trajectory"));
    actSpeed_->setText(actSpeed_->isChecked() ? Lang::s("speed_on") : Lang::s("speed_off"));
    actSpeed_->setToolTip(Lang::s("tip_speed"));
    actSkeleton_->setText(actSkeleton_->isChecked() ? Lang::s("skeleton_on") : Lang::s("skeleton_off"));
    actSkeleton_->setToolTip(Lang::s("tip_skeleton"));
    actStereo_->setText(actStereo_->isChecked() ? Lang::s("stereo_on") : Lang::s("stereo_off"));
    actStereo_->setToolTip(Lang::s("tip_stereo"));
    actCalibrate_->setText(Lang::s("calibration"));
    actCalibrate_->setToolTip(Lang::s("tip_calibrate"));
    actDepthOverlay_->setText(actDepthOverlay_->isChecked() ? Lang::s("depth_overlay_on") : Lang::s("depth_overlay_off"));
    actDepthOverlay_->setToolTip(Lang::s("tip_depth_overlay"));
    actStereoSettings_->setText(Lang::s("stereo_settings"));
    actCountLine_->setText(Lang::s("draw_line"));
    actCountLine_->setToolTip(Lang::s("tip_draw_line"));
    actClearLine_->setText(Lang::s("clear_line"));
    actClearLine_->setToolTip(Lang::s("tip_clear_line"));
    actLanguage_->setText(Lang::s("lang_toggle"));
    actLanguage_->setToolTip(Lang::s("tip_lang"));
    actFullScreen_->setText(Lang::s("menu_fullscreen"));
    actExit_->setText(Lang::s("menu_exit"));

    // Tab titles
    panelTabs_->setTabText(0, Lang::s("stats_title"));
    panelTabs_->setTabText(1, Lang::s("crossing_count"));
    panelTabs_->setTabText(2, Lang::s("pose_title"));
    panelTabs_->setTabText(3, Lang::s("depth_dock"));

    // Status bar
    deviceLabel_->setText(thread_.detector().isGpuEnabled()
        ? Lang::s("device_gpu") : Lang::s("device_cpu"));

    // Table headers
    statsTable_->setHorizontalHeaderLabels(
        {Lang::s("stats_class"), Lang::s("stats_count"), Lang::s("stats_unique")});
    totalUniqueLabel_->setText(Lang::s("stats_total_unique").arg(
        [] (const QMap<int,int>& m) { int t=0; for (auto v : m) t += v; return t; } (uniqueCounts_)));
    countingTable_->setHorizontalHeaderLabels(
        {Lang::s("stats_class"), Lang::s("forward"), Lang::s("reverse"), Lang::s("total")});
    poseTable_->setHorizontalHeaderLabels(
        {Lang::s("pose_keypoint"), Lang::s("pose_position"), Lang::s("pose_confidence")});
    depthTable_->setHorizontalHeaderLabels(
        {Lang::s("depth_track_id"), Lang::s("depth_class"),
         Lang::s("depth_dist"), Lang::s("depth_conf")});

    // Camera combo
    int curCam = cameraCombo_->currentData().toInt();
    cameraCombo_->blockSignals(true);
    cameraCombo_->clear();
    for (int i = 0; i < 10; i++)
        cameraCombo_->addItem(Lang::s("camera").arg(i), i);
    for (int i = 0; i < cameraCombo_->count(); i++) {
        if (cameraCombo_->itemData(i).toInt() == curCam) {
            cameraCombo_->setCurrentIndex(i);
            break;
        }
    }
    cameraCombo_->blockSignals(false);

    // Rebuild recent models submenu
    recentModelsMenu_->clear();
    for (const auto& m : std::as_const(recentModels_))
        recentModelsMenu_->addAction(QFileInfo(m).fileName(), [this, m]() { loadModelFile(m); });

    // Help menu
    auto actions = helpMenu_->actions();
    for (auto* a : std::as_const(actions))
        a->setText(Lang::s("about"));
}

void MainWindow::addRecentModel(const QString& path)
{
    recentModels_.removeAll(path);
    recentModels_.prepend(path);
    while (recentModels_.size() > 5)
        recentModels_.removeLast();
}
