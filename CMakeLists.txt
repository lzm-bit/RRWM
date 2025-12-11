cmake_minimum_required(VERSION 3.10)
project(RRWM_Cluster_Matching)

set(CMAKE_CXX_STANDARD 14)

# 查找依赖库
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

add_executable(rrwm_matching rrwm.cc)

# 链接库
target_link_libraries(rrwm_matching
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

