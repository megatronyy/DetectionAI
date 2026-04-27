#include "mainwindow.h"

#include <QApplication>
#include <QMap>

int main(int argc, char *argv[])
{
    qRegisterMetaType<QMap<int,int>>("QMap<int,int>");
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return QCoreApplication::exec();
}
