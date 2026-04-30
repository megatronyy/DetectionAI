#ifndef STEREOETTINGSDIALOG_H
#define STEREOETTINGSDIALOG_H

#include <QDialog>
#include <QComboBox>
#include <QSpinBox>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include "stereosource.h"
#include "stereomatcher.h"

class StereoSettingsDialog : public QDialog
{
    Q_OBJECT
public:
    explicit StereoSettingsDialog(const StereoSourceConfig& sourceConfig,
                                   const SGBMParams& sgbmParams,
                                   QWidget* parent = nullptr);

    StereoSourceConfig sourceConfig() const;
    SGBMParams sgbmParams() const;

private:
    QComboBox* hardwareCombo_;
    QSpinBox* leftCamSpin_;
    QSpinBox* rightCamSpin_;
    QLineEdit* leftRTSPEdit_;
    QLineEdit* rightRTSPEdit_;

    QSpinBox* blockSizeSpin_;
    QSpinBox* minDisparitySpin_;
    QSpinBox* numDisparitiesSpin_;
    QSpinBox* uniquenessSpin_;
    QSpinBox* speckleWindowSpin_;
    QSpinBox* speckleRangeSpin_;
    QDoubleSpinBox* baselineSpin_;
    QDoubleSpinBox* focalSpin_;
};

#endif // STEREOETTINGSDIALOG_H
