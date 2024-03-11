cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov8_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov8_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# rknn api
# set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)

# set(LIB_ARCH aarch64)
# set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

# opencv
find_package(OpenCV REQUIRED)

#rga
include_directories(/usr/include/rga)

set(CMAKE_INSTALL_RPATH "lib")

# rknn_yolov8_demo
include_directories( ${CMAKE_SOURCE_DIR}/include)

add_executable(rknn_yolov8_demo
        src/main.cc
        src/postprocess.cc
        src/model_inference.cpp
        src/idetector.cpp
)

target_link_libraries(rknn_yolov8_demo
  rknn_api
  ${RGA_LIB}
  ${OpenCV_LIBS}
  pthread
)


# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov8_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_yolov8_demo DESTINATION ./)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
