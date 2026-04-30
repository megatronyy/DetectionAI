#include "stereoettingsdialog.h"
#include "lang.h"
#include <QTabWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QDialogButtonBox>

StereoSettingsDialog::StereoSettingsDialog(const StereoSourceConfig& sourceConfig,
                                             const SGBMParams& sgbmParams,
                                             QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle(Lang::s("stereo_settings"));
    auto* tabs = new QTabWidget(this);

    // Hardware tab
    auto* hwWidget = new QWidget;
    auto* hwLayout = new QFormLayout(hwWidget);

    hardwareCombo_ = new QComboBox;
    hardwareCombo_->addItem(Lang::s("hw_dual_usb"), (int)StereoHardware::DualUSB);
    hardwareCombo_->addItem(Lang::s("hw_dual_rtsp"), (int)StereoHardware::DualRTSP);
    int hwIdx = hardwareCombo_->findData((int)sourceConfig.hardware);
    if (hwIdx >= 0) hardwareCombo_->setCurrentIndex(hwIdx);
    hwLayout->addRow(Lang::s("stereo_hw"), hardwareCombo_);

    leftCamSpin_ = new QSpinBox; leftCamSpin_->setRange(0, 20); leftCamSpin_->setValue(sourceConfig.leftCameraIndex);
    hwLayout->addRow(Lang::s("stereo_left"), leftCamSpin_);

    rightCamSpin_ = new QSpinBox; rightCamSpin_->setRange(0, 20); rightCamSpin_->setValue(sourceConfig.rightCameraIndex);
    hwLayout->addRow(Lang::s("stereo_right"), rightCamSpin_);

    leftRTSPEdit_ = new QLineEdit(QString::fromStdString(sourceConfig.leftRTSPUrl));
    hwLayout->addRow(Lang::s("stereo_left_url"), leftRTSPEdit_);

    rightRTSPEdit_ = new QLineEdit(QString::fromStdString(sourceConfig.rightRTSPUrl));
    hwLayout->addRow(Lang::s("stereo_right_url"), rightRTSPEdit_);

    tabs->addTab(hwWidget, Lang::s("stereo_hw"));

    // SGBM tab
    auto* sgbmWidget = new QWidget;
    auto* sgbmLayout = new QFormLayout(sgbmWidget);

    blockSizeSpin_ = new QSpinBox; blockSizeSpin_->setRange(3, 11); blockSizeSpin_->setSingleStep(2); blockSizeSpin_->setValue(sgbmParams.blockSize);
    sgbmLayout->addRow(Lang::s("sgbm_block_size"), blockSizeSpin_);

    minDisparitySpin_ = new QSpinBox; minDisparitySpin_->setRange(0, 100); minDisparitySpin_->setValue(sgbmParams.minDisparity);
    sgbmLayout->addRow(Lang::s("sgbm_min_disp"), minDisparitySpin_);

    numDisparitiesSpin_ = new QSpinBox; numDisparitiesSpin_->setRange(16, 256); numDisparitiesSpin_->setSingleStep(16); numDisparitiesSpin_->setValue(sgbmParams.numDisparities);
    sgbmLayout->addRow(Lang::s("sgbm_num_disp"), numDisparitiesSpin_);

    uniquenessSpin_ = new QSpinBox; uniquenessSpin_->setRange(0, 50); uniquenessSpin_->setValue(sgbmParams.uniquenessRatio);
    sgbmLayout->addRow(Lang::s("sgbm_uniqueness"), uniquenessSpin_);

    speckleWindowSpin_ = new QSpinBox; speckleWindowSpin_->setRange(0, 500); speckleWindowSpin_->setValue(sgbmParams.speckleWindowSize);
    sgbmLayout->addRow(Lang::s("sgbm_speckle_win"), speckleWindowSpin_);

    speckleRangeSpin_ = new QSpinBox; speckleRangeSpin_->setRange(0, 100); speckleRangeSpin_->setValue(sgbmParams.speckleRange);
    sgbmLayout->addRow(Lang::s("sgbm_speckle_range"), speckleRangeSpin_);

    baselineSpin_ = new QDoubleSpinBox; baselineSpin_->setRange(0.001, 1.0); baselineSpin_->setDecimals(3); baselineSpin_->setValue(sgbmParams.baselineMeters);
    baselineSpin_->setSuffix(" m");
    sgbmLayout->addRow(Lang::s("sgbm_baseline"), baselineSpin_);

    focalSpin_ = new QDoubleSpinBox; focalSpin_->setRange(1.0, 10000.0); focalSpin_->setDecimals(1); focalSpin_->setValue(sgbmParams.focalLengthPixels);
    focalSpin_->setSuffix(" px");
    sgbmLayout->addRow(Lang::s("sgbm_focal"), focalSpin_);

    tabs->addTab(sgbmWidget, Lang::s("sgbm_params"));

    // Buttons
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(tabs);
    mainLayout->addWidget(buttons);
}

StereoSourceConfig StereoSettingsDialog::sourceConfig() const
{
    StereoSourceConfig cfg;
    cfg.hardware = (StereoHardware)hardwareCombo_->currentData().toInt();
    cfg.leftCameraIndex = leftCamSpin_->value();
    cfg.rightCameraIndex = rightCamSpin_->value();
    cfg.leftRTSPUrl = leftRTSPEdit_->text().toStdString();
    cfg.rightRTSPUrl = rightRTSPEdit_->text().toStdString();
    return cfg;
}

SGBMParams StereoSettingsDialog::sgbmParams() const
{
    SGBMParams p;
    p.blockSize = blockSizeSpin_->value();
    p.minDisparity = minDisparitySpin_->value();
    p.numDisparities = numDisparitiesSpin_->value();
    p.uniquenessRatio = uniquenessSpin_->value();
    p.speckleWindowSize = speckleWindowSpin_->value();
    p.speckleRange = speckleRangeSpin_->value();
    p.baselineMeters = (float)baselineSpin_->value();
    p.focalLengthPixels = (float)focalSpin_->value();
    return p;
}
