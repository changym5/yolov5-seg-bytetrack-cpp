cmake_minimum_required(VERSION 3.5)
project(yolov5_seg)

set(ONNXRUNTIME_DIR ~/onnxruntime-linux-x64-gpu-1.14.1)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

include_directories(
    ${PROJECT_SOURCE_DIR}/bytetrack
    ${PROJECT_SOURCE_DIR}/yolo
    ${ONNXRUNTIME_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(
    yolov5_seg
    main.cpp
    
    yolo/yolov5_seg_onnx.cpp
    yolo/yolov5_seg_utils.cpp

    bytetrack/BytekalmanFilter.cpp
    bytetrack/ByteTrack.cpp
    bytetrack/lapjv.cpp
    bytetrack/STrack.cpp


)

target_link_libraries(
    yolov5_seg
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)