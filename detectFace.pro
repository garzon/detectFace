TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += c++11

SOURCES += main.cpp \

HEADERS += \
    FaceDetection.hpp \
    system_struct.hpp \


LIBS += /usr/lib/x86_64-linux-gnu/*.so \
/usr/local/lib/*.so

