#!/bin/bash -l
set -euo pipefail

# ----------------------------
# Environment setup
# ----------------------------
module purge
module load nvhpc
module load cuda
module load openmpi
module load llvm
module load cmake


# to support compiling on compute node
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

export CPM_SOURCE_CACHE="$HOME/.cache/CPM"

export NV_COMM_LIBS=$NVHPC_ROOT/Linux_x86_64/25.5/comm_libs

# NCCL
export NCCL_HOME=$NV_COMM_LIBS/nccl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NCCL_HOME/lib

# NVSHMEM
export NVSHMEM_HOME=$NV_COMM_LIBS/nvshmem
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVSHMEM_HOME/lib

# ----------------------------
# Config
# ----------------------------
BUILD_DIR="build"

# ----------------------------
# Functions
# ----------------------------

configure() {
    echo ">>> Configuring..."
    CXX=g++;cmake -S . -B "$BUILD_DIR"
}

build() {
    echo ">>> Building..."
    cmake --build "$BUILD_DIR" -j
}

clean() {
    echo ">>> Cleaning build directory..."
    rm -rf "$BUILD_DIR"
}

format() {
    echo ">>> Running clang-format..."
    cmake --build "$BUILD_DIR" -t fix-clang-format
}

usage() {
    echo "Usage: $0 {compile|clean|format}"
    echo
    echo "  compile   Configure (if needed) and build"
    echo "  clean     Remove build dir and rebuild"
    echo "  format    Run clang-format target"
    exit 1
}

# ----------------------------
# Main
# ----------------------------

if [[ $# -lt 1 ]]; then
    usage
fi

case "$1" in

    compile)
        [[ -d "$BUILD_DIR" ]] || configure
        build
        ;;

    clean)
        clean
        configure
        build
        ;;

    format)
        [[ -d "$BUILD_DIR" ]] || configure
        format
        ;;

    *)
        usage
        ;;
esac
