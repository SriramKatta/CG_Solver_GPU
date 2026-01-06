cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(test LANGUAGES CUDA C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


include(cmake/CPM.cmake)
include(cmake/CCCL.cmake)
include(cmake/NVTX.cmake)
include(cmake/GCXX.cmake)

include_directories(headers)
# include_directories(NVTX/c/include)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "80;86")

add_executable(exe-main main.cu)
target_link_libraries(exe-main PRIVATE CUDA::cudart)
target_link_libraries(exe-main PRIVATE CCCL::CUB)
target_link_libraries(exe-main PRIVATE CCCL::Thrust)
target_link_libraries(exe-main PRIVATE nvtx3-c)
target_link_libraries(exe-main PRIVATE gcxx::gcxx)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(exe-main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G -lineinfo>)
endif()

