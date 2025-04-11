cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(test LANGUAGES CUDA C)


include(cmake/CPM.cmake)
include(cmake/CCCL.cmake)
include(cmake/NVTX.cmake)

include_directories(headers)
include_directories(NVTX/c/include)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(exe-main main.cu)
target_link_libraries(exe-main PRIVATE CUDA::cudart)
target_link_libraries(exe-main PRIVATE CCCL::CUB)
target_link_libraries(exe-main PRIVATE CCCL::Thrust)
target_link_libraries(exe-main PRIVATE nvtx3-c)

#target_compile_options(exe-main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fast>)

