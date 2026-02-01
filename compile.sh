#!/bin/bash -l

module purge 
module load nvhpc
module load cuda
module load openmpi
module load cmake 

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

export CPM_SOURCE_CACHE=~/.cache/CPM/

export NV_COMM_LIBS=$NVHPC_ROOT/Linux_x86_64/25.5/comm_libs
#load nccl library 
export NCCL_HOME=$NV_COMM_LIBS/nccl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NCCL_HOME/lib
#load nvshmem library 
export NVSHMEM_HOME=$NV_COMM_LIBS/nvshmem
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVSHMEM_HOME/lib

cmake -S. -Bbuild 
cmake --build build -- -j


