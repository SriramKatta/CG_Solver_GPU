cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(test LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES "80;86")

set(CMAKE_BUILD_TYPE Release)

set(NVTX_MARK_OFF ON)

include(cmake/CPM.cmake)
include(cmake/CCCL.cmake)
include(cmake/NCCL.cmake)
include(cmake/NVTX.cmake)
include(cmake/GCXX.cmake)
include(cmake/FMT.cmake)
find_package(MPI REQUIRED COMPONENTS CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(headers)

add_executable(exe-main main.cu)
if(NVTX_MARK_OFF)
    target_compile_definitions(exe-main PUBLIC NVTX_DISABLE)
endif()
target_link_libraries(exe-main PRIVATE CCCL::CUB)
target_link_libraries(exe-main PRIVATE nvtx3-cpp)
target_link_libraries(exe-main PRIVATE gcxx::gcxx)
target_link_libraries(exe-main PRIVATE fmt::fmt)
target_link_libraries(exe-main PRIVATE MPI::MPI_CXX)
target_link_libraries(exe-main PRIVATE NCCL::NCCL)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(exe-main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G -lineinfo>)
endif()

