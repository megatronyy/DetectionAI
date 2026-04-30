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

private:
    QLabel* videoLabel_;
    QPushButton* pauseBtn_;
    QPushButton* screenshotBtn_;
    QPushButton* videoBtn_;
    QPushButton* recordBtn_;
    QPushButton* exportBtn_;
    QPushButton* networkCamBtn_;
    QPushButton* loopBtn_;
    QPushButton* switchModelBtn_;
    QPushButton* recentModelBtn_;
    QPushButton* classFilterBtn_;
    QPushButton* trackingBtn_;
    QPushButton* trajectoryBtn_;
    QPushButton* speedBtn_;
    QPushButton* skeletonBtn_;
    QPushButton* stereoBtn_;
    QPushButton* calibrateBtn_;
    QPushButton* depthOverlayBtn_;
    QPushButton* stereoSettingsBtn_;
    QPushButton* countLineBtn_;
    QPushButton* clearLineBtn_;
    QPushButton* langBtn_;
    QComboBox* cameraCombo_;
    QSlider* confSlider_;
    QLabel* confValueLabel_;
    QSlider* iouSlider_;
    QLabel* iouValueLabel_;
    QLabel* fpsLabel_;
    QLabel* detLabel_;
    QLabel* inferLabel_;
    QLabel* deviceLabel_;
    QDockWidget* statsDock_;
    QTableWidget* statsTable_;
    QPushButton* clearStatsBtn_;
    QPushButton* resetCountsBtn_;
    QLabel* totalUniqueLabel_;
    QMap<int, int> uniqueCounts_;

    QDockWidget* countingDock_;
    QTableWidget* countingTable_;
    QPushButton* clearCrossingBtn_;

    QDockWidget* poseDock_;
    QTableWidget* poseTable_;
    QLabel* posePersonLabel_;
    QSlider* kpConfSlider_;
    QLabel* kpConfValueLabel_;

    QDockWidget* depthDock_;
    QTableWidget* depthTable_;
    QDockWidget* pointCloudDock_;
    QLabel* pointCloudLabel_;

    enum class DrawMode { Idle, WaitingPt1, WaitingPt2 };
    DrawMode drawMode_ = DrawMode::Idle;
    cv::Point drawPt1_;

    InferenceThread thread_;
    QImage lastFrame_;
    bool paused_ = false;
    QString currentModelPath_;
    QSet<int> enabledClasses_;
    QMap<int, int> classStats_;
    QStringList recentModels_;
    StereoSourceConfig stereoConfig_;
    SGBMParams sgbmParams_;
    QString lastCalibPath_;

    void setupUI();
    void loadSettings();
    void saveSettings();
    void enumerateCameras();
    void refreshUIText();
    QPoint widgetToFrameCoords(const QPoint& widgetPos) const;
    void addRecentModel(const QString& path);
    void openVideoFile(const QString& path);
    void loadModelFile(const QString& path);
    void updateModelTypeUI();
};

#endif // MAINWINDOW_H
