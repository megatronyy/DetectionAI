#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QComboBox>
#include <QCloseEvent>
#include <QKeyEvent>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QDockWidget>
#include <QTableWidget>
#include <QTabWidget>
#include <QAction>
#include <QMenu>
#include <QInputDialog>
#include <QMouseEvent>
#include "inferencethread.h"
#include "stereosource.h"
#include "stereomatcher.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private slots:
    void onFrameReady(const QImage& image, int detCount, float fps,
                      float inferMs, const QMap<int,int>& classCounts);
    void onInputLost(const QString& msg);
    void onTogglePause();
    void onScreenshot();
    void onOpenVideo();
    void onCameraChanged(int index);
    void onConfChanged(int value);
    void onIouChanged(int value);
    void onToggleRecord();
    void onNetworkCamera();
    void onSwitchModel();
    void onClassFilter();
    void onToggleTracking(bool checked);
    void onToggleLoop(bool checked);
    void onToggleTrajectory(bool checked);
    void onToggleSpeed(bool checked);
    void onToggleSkeleton(bool checked);
    void onPoseDataUpdated(const std::vector<Detection>& dets);
    void onToggleStereo(bool checked);
    void onStereoSettings();
    void onCalibrate();
    void onToggleDepthOverlay(bool checked);
    void onDepthMapReady(const QImage& depthViz, float avgDepth);
    void onTrackingStatsUpdated(const QMap<int,int>& uniqueCounts, int totalUnique);
    void onDrawCountingLine();
    void onClearCountingLine();
    void onCrossingStatsUpdated(const QMap<int, QMap<int, int>>& counts);
    void onExport();
    void onToggleLanguage();
    void onAbout();
    void onClearStats();
    void onRecentModel();
    void onToggleFullScreen();

private:
    // Central display
    QLabel* videoLabel_;
    QComboBox* cameraCombo_;

    // Sliders
    QSlider* confSlider_;
    QLabel* confValueLabel_;
    QSlider* iouSlider_;
    QLabel* iouValueLabel_;

    // Status bar
    QLabel* fpsLabel_;
    QLabel* detLabel_;
    QLabel* inferLabel_;
    QLabel* deviceLabel_;

    // Menus
    QMenu* fileMenu_;
    QMenu* modelMenu_;
    QMenu* playbackMenu_;
    QMenu* trackingMenu_;
    QMenu* stereoMenu_;
    QMenu* viewMenu_;
    QMenu* helpMenu_;
    QMenu* recentModelsMenu_;

    // Actions
    QAction* actPause_;
    QAction* actScreenshot_;
    QAction* actRecord_;
    QAction* actExport_;
    QAction* actOpenVideo_;
    QAction* actNetworkCam_;
    QAction* actLoop_;
    QAction* actSwitchModel_;
    QAction* actClassFilter_;
    QAction* actTracking_;
    QAction* actTrajectory_;
    QAction* actSpeed_;
    QAction* actSkeleton_;
    QAction* actStereo_;
    QAction* actDepthOverlay_;
    QAction* actCalibrate_;
    QAction* actStereoSettings_;
    QAction* actCountLine_;
    QAction* actClearLine_;
    QAction* actLanguage_;
    QAction* actFullScreen_;
    QAction* actExit_;

    // Unified right panel
    QDockWidget* panelDock_;
    QTabWidget* panelTabs_;
    QWidget* statsPage_;
    QWidget* trackingPage_;
    QWidget* posePage_;
    QWidget* depthPage_;

    // Tables (inside tabs)
    QTableWidget* statsTable_;
    QTableWidget* countingTable_;
    QTableWidget* poseTable_;
    QTableWidget* depthTable_;

    // Stats page widgets
    QLabel* totalUniqueLabel_;

    // Pose page widgets
    QLabel* posePersonLabel_;
    QSlider* kpConfSlider_;
    QLabel* kpConfValueLabel_;

    // Depth page widgets
    QLabel* pointCloudLabel_;

    // Draw mode
    enum class DrawMode { Idle, WaitingPt1, WaitingPt2 };
    DrawMode drawMode_ = DrawMode::Idle;
    cv::Point drawPt1_;

    // State
    InferenceThread thread_;
    QImage lastFrame_;
    bool paused_ = false;
    QString currentModelPath_;
    QSet<int> enabledClasses_;
    QMap<int, int> classStats_;
    QMap<int, int> uniqueCounts_;
    QStringList recentModels_;
    StereoSourceConfig stereoConfig_;
    SGBMParams sgbmParams_;
    QString lastCalibPath_;

    void setupUI();
    void loadSettings();
    void saveSettings();
    void enumerateCameras();
    void refreshUIText();
    void updateModelTypeUI();
    QPoint widgetToFrameCoords(const QPoint& widgetPos) const;
    void addRecentModel(const QString& path);
    void openVideoFile(const QString& path);
    void loadModelFile(const QString& path);
    QAction* createAction(const QString& text, const QString& tip,
                          const QKeySequence& shortcut = QKeySequence(),
                          bool checkable = false, bool checked = false);
};

#endif // MAINWINDOW_H
