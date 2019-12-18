cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch)

set (CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


find_package( OpenCV REQUIRED )

add_executable(pytorch main.cpp infer.cpp ./utils/opencvutils.cpp ./utils/torchutils.cpp)
target_link_libraries(pytorch "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
set_property(TARGET pytorch PROPERTY CXX_STANDARD 14)

