#pragma once

#include <gcxx/api.hpp>

#include <cub/cub.cuh>

constexpr auto blockSize_x = 32, blockSize_y = 16;


template <typename VT>
using restrict_mdspan =
  gcxx::mdspan<VT, gcxx::dextents<int, 2>, gcxx::layout_right,
               gcxx::restrict_accessor<VT>>;

template <typename VT>
__global__ void applystencil(const VT *const __restrict__ p,
                             VT *__restrict__ ap, const size_t nx,
                             const size_t ny) {
  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY)
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      ap[j * nx + i] =
        4 * p[j * nx + i] - (p[j * nx + i - 1] + p[j * nx + i + 1] +
                             p[(j - 1) * nx + i] + p[(j + 1) * nx + i]);
    }
}

template <typename VT>
__global__ void cgUpdateSol(const VT *const __restrict__ p, VT *__restrict__ u,
                            const VT alpha, const size_t nx, const size_t ny) {
  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY)
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      u[j * nx + i] += alpha * p[j * nx + i];
    }
}

template <typename VT>
__global__ void cgUpdateRes(const VT *const __restrict__ ap,
                            VT *__restrict__ res, const VT alpha,
                            const size_t nx, const size_t ny) {
  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY)
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      res[j * nx + i] = res[j * nx + i] - alpha * ap[j * nx + i];
    }
}

template <typename VT>
__global__ void cgUpdateP(VT beta, const VT *const __restrict__ res,
                          VT *__restrict__ p, size_t nx, size_t ny) {
  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY)
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      p[j * nx + i] = res[j * nx + i] + beta * p[j * nx + i];
    }
}

template <typename VT>
__global__ void residual_initp(VT *__restrict__ res, VT *__restrict__ p,
                               const VT *const __restrict__ rhs,
                               const VT *const __restrict__ u, size_t nx,
                               size_t ny) {

  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY)
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      auto temp =
        rhs[j * nx + i] -
        (4 * u[j * nx + i] - (u[j * nx + i - 1] + u[j * nx + i + 1] +
                              u[(j - 1) * nx + i] + u[(j + 1) * nx + i]));
      res[j * nx + i] = temp;
      p[j * nx + i] = temp;
    }
}


template <typename VT>
__device__ VT innerproduct_tile(const VT *A, const VT *B, size_t nx,
                                size_t ny) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ix = blockIdx.x * blockDim.x + tx + 1;
  const int iy = blockIdx.y * blockDim.y + ty + 1;
  const int strideX = blockDim.x * gridDim.x;
  const int strideY = blockDim.y * gridDim.y;
  VT sum = static_cast<VT>(0);

  for (int y = iy; y < ny - 1; y += strideY) {
    for (int x = ix; x < nx - 1; x += strideX) {
      int idx = y * nx + x;
      sum += A[idx] * B[idx];
    }
  }
  return sum;
}


template <typename VT>
__global__ void innerproduct(const VT *const __restrict__ A,
                             const VT *const __restrict__ B, VT *result,
                             size_t nx, size_t ny) {

  using blockreduce =
    cub::BlockReduce<VT, blockSize_x,
                     cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS,
                     blockSize_y>;

  __shared__ typename blockreduce::TempStorage temp_storage;

  auto thread_sum = innerproduct_tile(A, B, nx, ny);

  auto aggregate = blockreduce(temp_storage).Sum(thread_sum);

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd(result, aggregate);
  }
}

template <typename VT>
__global__ void devide_kernel(VT *res, VT *Numerator, VT *Denominator) {
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix == 0 && iy == 0) {
    *res = (*Numerator) / (*Denominator);
  }
}

template <typename VT>
VT resnormsqcalc(const VT *const __restrict__ res, size_t nx, size_t ny,
                 dim3 numblocks, dim3 blocksize) {
  size_t smemsize = blocksize.x * blocksize.y * sizeof(VT);
  auto ressqnorm_raii = gcxx::memory::make_device_managed_unique_ptr<VT>(1);
  gcxx::memory::Memset(ressqnorm_raii, 0, 1);
  VT *ressqnorm = ressqnorm_raii.get();
  VT hostressqnorm = 0.0;
  innerproduct<<<numblocks, blocksize, smemsize>>>(res, res, ressqnorm, nx, ny);
  gcxx::memory::Copy(&hostressqnorm, ressqnorm, 1);
  return hostressqnorm;
}

template <typename VT>
VT alphadencalc(const VT *const __restrict__ p, const VT *const __restrict__ ap,
                size_t nx, size_t ny, dim3 numblocks, dim3 blocksize) {
  size_t smemsize = blocksize.x * blocksize.y * sizeof(VT);
  auto alphaden_raii = gcxx::memory::make_device_unique_ptr<VT>(1);
  gcxx::memory::Memset(alphaden_raii, 0, 1);
  VT *alphaden = alphaden_raii.get();
  VT hostalphaden = 1.0;
  innerproduct<<<numblocks, blocksize, smemsize>>>(p, ap, alphaden, nx, ny);
  gcxx::memory::Copy(&hostalphaden, alphaden, 1);
  return hostalphaden;
}