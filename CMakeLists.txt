cmake_minimum_required(VERSION 3.0)
project(Bytetrack)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

set(ONNXRUNTIME_DIR ~/onnxruntime-linux-x64-gpu-1.14.1)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/bytetrack
    ${PROJECT_SOURCE_DIR}/yolo
    ${ONNXRUNTIME_DIR}/include
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(Bytetrack SHARED
    yolo/yolov5_seg_onnx.cpp
    yolo/yolov5_seg_utils.cpp

    bytetrack/BytekalmanFilter.cpp
    bytetrack/ByteTrack.cpp
    bytetrack/lapjv.cpp
    bytetrack/STrack.cpp
)

target_link_libraries(Bytetrack
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

add_executable(demo
    main.cpp
)

target_link_libraries(demo ${PROJECT_NAME})