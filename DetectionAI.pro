QT += widgets

CONFIG += c++17

# OpenCV 路径 自行修改
OPENCV_INC = C:/Users/twfx7/opencv/build/include
OPENCV_LIB = C:/Users/twfx7/opencv/build/x64/vc16/lib

# ONNX Runtime 路径 自行修改
ORT_INC    = C:/Users/twfx7/onnxruntime/include
ORT_LIB    = C:/Users/twfx7/onnxruntime/lib

INCLUDEPATH += $$OPENCV_INC
INCLUDEPATH += $$ORT_INC

CONFIG(debug, debug|release): LIBS += -L$$OPENCV_LIB -lopencv_world4130d
CONFIG(release, debug|release): LIBS += -L$$OPENCV_LIB -lopencv_world4130
LIBS += -L$$ORT_LIB -lonnxruntime

# GPU (CUDA) support - uncomment the following line to enable:
# DEFINES += USE_CUDA

SOURCES += main.cpp \
           mainwindow.cpp \
           yolodetector.cpp \
           inferencethread.cpp

HEADERS  += mainwindow.h \
            yolodetector.h \
            inferencethread.h

FORMS    += mainwindow.ui
RC_FILE  = app.rc

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
