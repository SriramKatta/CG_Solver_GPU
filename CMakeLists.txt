cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(test LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES "80;86")

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

set(NVTX_MARK_OFF ON)

include(cmake/CPM.cmake)
include(cmake/CCCL.cmake)
include(cmake/NCCL.cmake)
include(cmake/NVTX.cmake)
include(cmake/GCXX.cmake)
include(cmake/FMT.cmake)
include(cmake/argparse.cmake)
find_package(MPI REQUIRED COMPONENTS CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(headers)

set(EXE_NAME cg_solver)

add_executable(${EXE_NAME} main.cu)
if(NVTX_MARK_OFF)
    target_compile_definitions(${EXE_NAME} PUBLIC NVTX_DISABLE)
endif()
target_link_libraries(${EXE_NAME} PRIVATE CCCL::CUB)
target_link_libraries(${EXE_NAME} PRIVATE nvtx3-cpp)
target_link_libraries(${EXE_NAME} PRIVATE gcxx::gcxx)
target_link_libraries(${EXE_NAME} PRIVATE fmt::fmt)
target_link_libraries(${EXE_NAME} PRIVATE MPI::MPI_CXX)
target_link_libraries(${EXE_NAME} PRIVATE NCCL::NCCL)
target_link_libraries(${EXE_NAME} PRIVATE argparse::argparse)


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${EXE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G -lineinfo>)
endif()

add_subdirectory(tests)