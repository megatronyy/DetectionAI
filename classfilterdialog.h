#ifndef CLASSFILTERDIALOG_H
#define CLASSFILTERDIALOG_H

#include <QDialog>
#include <QCheckBox>
#include <QSet>

class ClassFilterDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ClassFilterDialog(const QSet<int>& enabledClasses,
                               QWidget* parent = nullptr);
    QSet<int> selectedClasses() const;

private:
    QList<QCheckBox*> checkBoxes_;
    void onSelectAll();
    void onSelectNone();
};

#endif // CLASSFILTERDIALOG_H
