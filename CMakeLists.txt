cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(test LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES "80;86")


include(cmake/CPM.cmake)
include(cmake/CCCL.cmake)
include(cmake/NVTX.cmake)
include(cmake/GCXX.cmake)
include(cmake/FMT.cmake)

include_directories(headers)

add_executable(exe-main main.cu)
target_link_libraries(exe-main PRIVATE CCCL::CUB)
target_link_libraries(exe-main PRIVATE nvtx3-cpp)
target_link_libraries(exe-main PRIVATE gcxx::gcxx)
target_link_libraries(exe-main PRIVATE fmt::fmt)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(exe-main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G -lineinfo>)
endif()

