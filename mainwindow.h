#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QComboBox>
#include <QCloseEvent>
#include <QKeyEvent>
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

private slots:
    void onFrameReady(const QImage& image, int detCount, float fps);
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

private:
    QLabel* videoLabel_;
    QPushButton* pauseBtn_;
    QPushButton* screenshotBtn_;
    QPushButton* videoBtn_;
    QPushButton* recordBtn_;
    QPushButton* networkCamBtn_;
    QPushButton* switchModelBtn_;
    QPushButton* classFilterBtn_;
    QPushButton* trackingBtn_;
    QComboBox* cameraCombo_;
    QSlider* confSlider_;
    QLabel* confValueLabel_;
    QSlider* iouSlider_;
    QLabel* iouValueLabel_;
    QLabel* fpsLabel_;
    QLabel* detLabel_;
    QLabel* deviceLabel_;

    InferenceThread thread_;
    QImage lastFrame_;
    bool paused_ = false;
    QString currentModelPath_;
    QSet<int> enabledClasses_;

    void setupUI();
    void loadSettings();
    void saveSettings();
    void restartThread();
};

#endif // MAINWINDOW_H
