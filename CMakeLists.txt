cmake_minimum_required(VERSION 2.8.3)
project(calib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/third_lib/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_compile_options(-std=c++14)
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(Ceres_DIR /home/ylh/software/ceres-solver-1.14.0/install/lib/cmake/Ceres)
# set(Ceres_DIR /home/ylh/software/ceres-solver/install/lib/cmake/Ceres)

find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARY_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PCL REQUIRED)
link_directories(${PCL_LIBRARY_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
link_directories(${Ceres_LIBRARY_DIRS})

include_directories("/usr/include/eigen3")
include_directories("/usr/include/jsoncpp")

include_directories(
	${PROJECT_SOURCE_DIR}/include
)
link_directories(
  ${PROJECT_SOURCE_DIR}/third_lib/lib/
)
FILE(GLOB SHARED_LIBS ${PROJECT_SOURCE_DIR}/third_lib/lib/*)

add_executable(calib_lidar2camera src/main_lidar2camera.cpp)
target_link_libraries(calib_lidar2camera
                      ${PCL_LIBRARIES}
                      ${OpenCV_LIBS}
                      ${CERES_LIBRARIES}
                      libjsoncpp.a
                      )                 