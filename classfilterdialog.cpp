#include "classfilterdialog.h"
#include "yolodetector.h"
#include "lang.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QPushButton>
#include <QDialogButtonBox>

ClassFilterDialog::ClassFilterDialog(const QSet<int>& enabledClasses,
                                     QWidget* parent)
    : QDialog(parent)
{
    setWindowTitle(Lang::s("class_filter_title"));
    resize(320, 500);

    auto* mainLayout = new QVBoxLayout(this);

    // Select all / none buttons
    auto* btnLayout = new QHBoxLayout;
    auto* allBtn = new QPushButton(Lang::s("select_all"));
    auto* noneBtn = new QPushButton(Lang::s("select_none"));
    btnLayout->addWidget(allBtn);
    btnLayout->addWidget(noneBtn);
    mainLayout->addLayout(btnLayout);

    connect(allBtn, &QPushButton::clicked, this, &ClassFilterDialog::onSelectAll);
    connect(noneBtn, &QPushButton::clicked, this, &ClassFilterDialog::onSelectNone);

    // Scroll area with checkboxes
    auto* scrollArea = new QScrollArea;
    scrollArea->setWidgetResizable(true);
    auto* container = new QWidget;
    auto* checkLayout = new QVBoxLayout(container);
    checkLayout->setSpacing(2);

    bool allEnabled = enabledClasses.isEmpty();
    for (int i = 0; i < YOLODetector::NUM_CLASSES; i++) {
        QString text = QString("%1: %2").arg(i).arg(
            QString::fromStdString(YOLODetector::CLASS_NAMES[i]));
        auto* cb = new QCheckBox(text);
        cb->setChecked(allEnabled || enabledClasses.contains(i));
        checkBoxes_.append(cb);
        checkLayout->addWidget(cb);
    }
    checkLayout->addStretch();
    scrollArea->setWidget(container);
    mainLayout->addWidget(scrollArea);

    // Ok / Cancel
    auto* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    mainLayout->addWidget(buttonBox);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

QSet<int> ClassFilterDialog::selectedClasses() const
{
    QSet<int> result;
    for (int i = 0; i < checkBoxes_.size(); i++) {
        if (checkBoxes_[i]->isChecked())
            result.insert(i);
    }
    return result;
}

void ClassFilterDialog::onSelectAll()
{
    for (auto* cb : checkBoxes_) cb->setChecked(true);
}

void ClassFilterDialog::onSelectNone()
{
    for (auto* cb : checkBoxes_) cb->setChecked(false);
}
