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
#include "inferencethread.h"

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

    InferenceThread thread_;
    QImage lastFrame_;
    bool paused_ = false;
    QString currentModelPath_;
    QSet<int> enabledClasses_;
    QMap<int, int> classStats_;
    QStringList recentModels_;

    void setupUI();
    void loadSettings();
    void saveSettings();
    void enumerateCameras();
    void refreshUIText();
    void addRecentModel(const QString& path);
    void openVideoFile(const QString& path);
    void loadModelFile(const QString& path);
};

#endif // MAINWINDOW_H
