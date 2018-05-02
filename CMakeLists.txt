cmake_minimum_required(VERSION 3.9)
project(ObjectsDetector)

set(CMAKE_CXX_STANDARD 11)
include_directories( /usr/local/include/opencv2 )
add_executable(ObjectsDetector main.cpp)

find_package(OpenCV REQUIRED core highgui imgproc objdetect dnn)

target_link_libraries( ObjectsDetector ${OpenCV_LIBS} )