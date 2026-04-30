#include "mainwindow.h"
#include "yolodetector.h"

#include <QApplication>
#include <QMap>
#include <vector>

int main(int argc, char *argv[])
{
    qRegisterMetaType<QMap<int,int>>("QMap<int,int>");
    qRegisterMetaType<std::vector<Detection>>("std::vector<Detection>");
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return QCoreApplication::exec();
}
