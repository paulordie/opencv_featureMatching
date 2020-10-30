cmake_minimum_required(VERSION 3.17)
project(featureMatching)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(OpenCV_FOUND 1)
find_package( OpenCV REQUIRED )

set(SOURCE_FILES main.cpp)

add_executable(featureMatching ${SOURCE_FILES})

target_link_libraries(featureMatching ${OpenCV_LIBS})