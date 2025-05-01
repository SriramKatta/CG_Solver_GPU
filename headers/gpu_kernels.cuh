#pragma once


template <class VT>
struct SharedMemory {
  __device__ inline operator VT *() {
    extern __shared__ int __smem[];
    return (VT *)__smem;
  }

  __device__ inline operator const VT *() const {
    extern __shared__ int __smem[];
    return (VT *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <typename VT>
__global__ void cgAp(const VT *const __restrict__ p, VT *__restrict__ ap,
                     const size_t nx, const size_t ny) {
  size_t tx = threadIdx.x;
  size_t ty = threadIdx.y;
  size_t i = blockDim.x * blockIdx.x + tx + 1;
  size_t j = blockDim.y * blockIdx.y + ty + 1;

  if (i >= nx - 1 || j >= ny - 1)
    return;

  size_t globalIdx = j * nx + i;
  size_t smem_pitch = blockDim.x + 2;
  size_t localIdx = (ty + 1) * smem_pitch + (tx + 1);

  // load data into shared mem
  VT *tile = SharedMemory<VT>();
  tile[localIdx] = p[globalIdx];
  if (0 == tx) {
    tile[localIdx - 1] = p[globalIdx - 1];
  }
  if (blockDim.x - 1 == tx) {
    tile[localIdx + 1] = p[globalIdx + 1];
  }

  if (0 == ty) {
    tile[localIdx - smem_pitch] = p[globalIdx - nx];
  }
  if (blockDim.y - 1 == ty) {
    tile[localIdx + smem_pitch] = p[globalIdx + nx];
  }
  __syncthreads();

  ap[j * nx + i] =
    4 * p[j * nx + i] - (tile[localIdx - 1] + tile[localIdx + 1] +
                         tile[localIdx - smem_pitch] + tile[localIdx + smem_pitch]);
}

template <typename VT>
inline void cgApcalc(const dim3 &numBlocks, const dim3 &blockSize,
                     VT *__restrict__ &p, VT *__restrict__ &ap,
                     const size_t &nx, const size_t &ny) {
  size_t totalthreads = (blockSize.x + 2) * (blockSize.y + 2) * blockSize.z;
  dim3 numBlockslocal((nx + blockSize.x - 1) / blockSize.x,
                      (ny + blockSize.y - 1) / blockSize.y, 1);
  cgAp<VT><<<numBlockslocal, blockSize, totalthreads * sizeof(VT)>>>(p, ap,
                                                                        nx, ny);
  checkCudaError(cudaDeviceSynchronize());
}
