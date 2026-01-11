#pragma once


#include <nvtx3/nvtx3.hpp>

#include <fmt/format.h>
#include <cub/block/block_reduce.cuh>
#include <gcxx/api.hpp>


constexpr auto blockSize_x = 32, blockSize_y = 16;

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
                            const VT *alpha, const size_t nx, const size_t ny) {
  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY)
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      u[j * nx + i] += (*alpha) * p[j * nx + i];
    }
}

template <typename VT>
__global__ void cgUpdateRes(const VT *const __restrict__ ap,
                            VT *__restrict__ res, const VT *alpha,
                            const size_t nx, const size_t ny) {
  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY)
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      res[j * nx + i] = res[j * nx + i] - (*alpha) * ap[j * nx + i];
    }
}

template <typename VT>
__global__ void cgUpdateP(const VT *beta, const VT *const __restrict__ res,
                          VT *__restrict__ p, size_t nx, size_t ny) {
  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY)
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      p[j * nx + i] = res[j * nx + i] + (*beta) * p[j * nx + i];
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
__device__ VT innerproduct_tile(const VT *__restrict__ A,
                                const VT *__restrict__ B, size_t nx,
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
  using block_reduce =
    cub::BlockReduce<VT, blockSize_x, cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                     blockSize_y>;

  __shared__ typename block_reduce::TempStorage tempstore;

  auto thread_data = innerproduct_tile(A, B, nx, ny);

  auto blocksum = block_reduce(tempstore).Sum(thread_data);

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd(result, blocksum);
  }
}

template <typename VT>
__global__ void gpu_devide(const VT *const num, const VT *const den, VT *res) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *res = (*num) / (*den);
  }
}

template <typename VT>
void resnormsqcalc(const VT *const __restrict__ res, size_t nx, size_t ny,
                   dim3 numblocks, dim3 blocksize, VT *ressqnorm,
                   gcxx::StreamView sv) {
  size_t smemsize = blocksize.x * blocksize.y * sizeof(VT);
  gcxx::memory::Memset(ressqnorm, 0, 1, sv);
  innerproduct<<<numblocks, blocksize, smemsize, sv>>>(res, res, ressqnorm, nx,
                                                       ny);
}

template <typename VT>
void alphacalc(const VT *const __restrict__ p, const VT *const __restrict__ ap,
               size_t nx, size_t ny, dim3 numblocks, dim3 blocksize,
               VT *alphanum, VT *alpha, gcxx::StreamView sv) {
  size_t smemsize = blocksize.x * blocksize.y * sizeof(VT);
  auto alphaden_raii = gcxx::memory::make_device_unique_ptr<VT>(1, sv);
  gcxx::memory::Memset(alphaden_raii, 0, 1, sv);
  VT *alphaden = alphaden_raii.get();
  innerproduct<<<numblocks, blocksize, smemsize, sv>>>(p, ap, alphaden, nx, ny);
  gpu_devide<<<1, 1, 0, sv>>>(alphanum, alphaden, alpha);
}

template <typename VT>
void core_CG(dim3 &numBlocks, dim3 &blockSize, gcxx::v1::Stream &str1,
             VT *__restrict__ &p, VT *__restrict__ &ap, const size_t &nx,
             const size_t &ny, VT *&curResSq, VT *&alpha, VT *__restrict__ &u,
             VT *__restrict__ &res, VT *&nextResSq, VT *&beta);

template <typename VT>
inline size_t conjugateGradient(const VT *const __restrict__ rhs,
                                VT *__restrict__ u, VT *__restrict__ res,
                                VT *__restrict__ p, VT *__restrict__ ap,
                                const size_t nx, const size_t ny,
                                const size_t maxIt) {

  dim3 blockSize(blockSize_x, blockSize_y);
  int smcount = 0;
  cudaDeviceGetAttribute(&smcount, cudaDevAttrMultiProcessorCount, 0);
  dim3 numBlocks(smcount, 10);

  residual_initp<VT><<<numBlocks, blockSize>>>(res, p, rhs, u, nx, ny);

  auto nextResSq_raii = gcxx::memory::make_device_managed_unique_ptr<VT>(1);

  auto mempoolhand = gcxx::MemPoolView::GetDefaultMempool(gcxx::Device::get());
  mempoolhand.SetReleaseThreshold(std::numeric_limits<uint64_t>::max());

  auto curResSq_raii = gcxx::memory::make_device_unique_ptr<VT>(1);
  auto alpha_raii = gcxx::memory::make_device_unique_ptr<VT>(1);
  auto beta_raii = gcxx::memory::make_device_unique_ptr<VT>(1);
  gcxx::memory::Memset(curResSq_raii, 0, 1);
  gcxx::memory::Memset(nextResSq_raii, 0, 1);
  gcxx::memory::Memset(alpha_raii, 0, 1);
  gcxx::memory::Memset(beta_raii, 0, 1);

  gcxx::Stream str1(gcxx::flags::streamType::SyncWithNull);

  VT *curResSq = curResSq_raii.get();
  VT *nextResSq = nextResSq_raii.get();
  VT *alpha = alpha_raii.get();
  VT *beta = beta_raii.get();

  // compute residual norm
  resnormsqcalc(res, nx, ny, numBlocks, blockSize, curResSq, str1);

  bool isgraphbuilt = false;
  gcxx::GraphExec exec;

  // main loop
  for (size_t it = 0; it < maxIt; ++it) {
    nvtx3::scoped_range loop{"main loop"};

    if (!isgraphbuilt) {
      gcxx::Graph graph;
      str1.BeginCaptureToGraph(graph, gcxx::flags::streamCaptureMode::Global);
      core_CG(numBlocks, blockSize, str1, p, ap, nx, ny, curResSq, alpha, u,
              res, nextResSq, beta);
      str1.EndCaptureToGraph(graph);
      exec = graph.Instantiate();
      isgraphbuilt = true;
    }

    exec.Launch(str1);

    // check exit criterion
    cudaMemPrefetchAsync(nextResSq, sizeof(VT), cudaCpuDeviceId, str1);
    str1.Synchronize();  // Needed
    if (sqrt(*nextResSq) <= 1e-12) {
      return it;
    }
    if (0 == it % 100)
      fmt::print("     {} : {}\n", it, sqrt(*nextResSq));
  }

  return maxIt;
}

template <typename VT>
inline void core_CG(dim3 &numBlocks, dim3 &blockSize, gcxx::v1::Stream &str1,
                    VT *__restrict__ &p, VT *__restrict__ &ap, const size_t &nx,
                    const size_t &ny, VT *&curResSq, VT *&alpha,
                    VT *__restrict__ &u, VT *__restrict__ &res, VT *&nextResSq,
                    VT *&beta) {
  // compute A * p
  nvtxRangePushA("Ap");
  applystencil<VT><<<numBlocks, blockSize, 0, str1>>>(p, ap, nx, ny);
  nvtxRangePop();

  nvtxRangePushA("alpha");
  alphacalc(p, ap, nx, ny, numBlocks, blockSize, curResSq, alpha, str1);
  nvtxRangePop();

  // update solution
  nvtxRangePushA("solution");
  cgUpdateSol<VT><<<numBlocks, blockSize, 0, str1>>>(p, u, alpha, nx, ny);
  nvtxRangePop();

  // update residual
  nvtxRangePushA("residual");
  cgUpdateRes<VT><<<numBlocks, blockSize, 0, str1>>>(ap, res, alpha, nx, ny);
  nvtxRangePop();

  // compute residual norm
  nvtxRangePushA("resNorm");
  resnormsqcalc(res, nx, ny, numBlocks, blockSize, nextResSq, str1);
  nvtxRangePop();

  // compute beta
  nvtxRangePushA("beta");
  // cudaMemPrefetchAsync(nextResSq, sizeof(VT), cudaCpuDeviceId);
  gpu_devide<<<1, 1, 0, str1>>>(nextResSq, curResSq, beta);
  // cudaMemcpy(curResSq, nextResSq, sizeof(VT), cudaMemcpyDeviceToDevice);
  gcxx::memory::Copy(curResSq, nextResSq, 1, str1);
  nvtxRangePop();

  // update p
  nvtxRangePushA("p");
  cgUpdateP<<<numBlocks, blockSize, 0, str1>>>(beta, res, p, nx, ny);
  nvtxRangePop();
}
