#pragma once


#include <nvtx3/nvtx3.hpp>

#include <fmt/format.h>
#include <cub/block/block_reduce.cuh>
#include <gcxx/api.hpp>

#include <mpi_comm.hpp>
#include <nccl_comm.hpp>


// TODO : modify all kernels to respect boundaries and work accordingly
// TODO : modify all kernels to respect boundaries and work accordingly
// TODO : modify all kernels to respect boundaries and work accordingly
// TODO : modify all kernels to respect boundaries and work accordingly
// TODO : modify all kernels to respect boundaries and work accordingly
// TODO : modify all kernels to respect boundaries and work accordingly
// TODO : modify all kernels to respect boundaries and work accordingly
// TODO : modify all kernels to respect boundaries and work accordingly
// TODO : modify all kernels to respect boundaries and work accordingly


constexpr auto blockSize_x = 32, blockSize_y = 16;

template <typename VT>
__global__ void applystencil(const VT *const __restrict__ p,
                             VT *__restrict__ ap, const size_t nx,
                             const size_t iy_start, const size_t iy_end) {
  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < iy_end; j += gridStrideY)
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      ap[j * nx + i] =
        4 * p[j * nx + i] - (p[j * nx + i - 1] + p[j * nx + i + 1] +
                             p[(j - 1) * nx + i] + p[(j + 1) * nx + i]);
    }
}

template <typename VT>
void launch_apply_stencil(gcxx::StreamView str1l, gcxx::StreamView str1h,
                          dim3 numBlocks, dim3 blockSize, VT *__restrict__ &p,
                          VT *__restrict__ &ap, size_t nx, size_t ny) {
  // needed since all streams are indepenedent  and may result in unconnected graph nodes
  str1h.WaitOnEvent(str1l.RecordEvent(gcxx::flags::eventCreate::disableTiming));

  // external edges // to implement edge comm
  gcxx::launch::Kernel(str1h, numBlocks, blockSize, 0, applystencil<VT>, p, ap,
                       nx, 0, 1);
  gcxx::launch::Kernel(str1h, numBlocks, blockSize, 0, applystencil<VT>, p, ap,
                       nx, ny - 2, ny - 1);

  // internal block
  gcxx::launch::Kernel(str1l, numBlocks, blockSize, 0, applystencil<VT>, p, ap,
                       nx, 1, ny - 2);
  str1l.WaitOnEvent(str1h.RecordEvent(gcxx::flags::eventCreate::disableTiming));
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
void launch_resnormsqcalc(const VT *const __restrict__ res, size_t nx,
                          size_t ny, dim3 numblocks, dim3 blocksize,
                          VT *ressqnorm, gcxx::StreamView sv,
                          ncclcommview ncomm) {
  size_t smemsize = blocksize.x * blocksize.y * sizeof(VT);
  gcxx::memory::Memset(ressqnorm, 0, 1, sv);
  gcxx::launch::Kernel(sv, numblocks, blocksize, smemsize, innerproduct<VT>,
                       res, res, ressqnorm, nx, ny);
  ncomm.allreduce(ressqnorm, ressqnorm, 1, ncclSum, sv.getRawStream());
}

template <typename VT>
void launch_alphacalc(const VT *const __restrict__ p,
                      const VT *const __restrict__ ap, size_t nx, size_t ny,
                      dim3 numblocks, dim3 blocksize, VT *alphanum,
                      VT *alphaden, VT *alpha, gcxx::StreamView sv,
                      ncclcommview ncomm) {
  size_t smemsize = blocksize.x * blocksize.y * sizeof(VT);
  gcxx::memory::Memset(alphaden, 0, 1, sv);
  gcxx::launch::Kernel(sv, numblocks, blocksize, smemsize, innerproduct<VT>, p,
                       ap, alphaden, nx, ny);
  ncomm.allreduce(alphaden, alphaden, 1, ncclSum, sv);
  gcxx::launch::Kernel(sv, 1, 1, 0, gpu_devide<VT>, alphanum, alphaden, alpha);
}

template <typename VT>
__global__ void cond_kernel(gcxx::deviceGraphConditionalHandle_t hand,
                            VT *res_sq, size_t *iter, size_t maxiter) {
  if (threadIdx.x != 0)
    return;
  ++(*iter);
#if defined(DEBUG)
  if ((*iter) % 100 == 0) {
    printf("  %ld : %f\n", *iter, sqrt(*res_sq));
  }
#endif
  // Check if we should stop
  if (sqrt(*res_sq) < 1e-12 || *iter >= maxiter) {
    gcxx::Graph::SetConditional(hand, 0);  // Stop the loop
  }
}

template <typename VT>
inline void core_CG(dim3 &numBlocks, dim3 &blockSize, VT *__restrict__ &p,
                    VT *__restrict__ &ap, const size_t &nx, const size_t &ny,
                    VT *&curResSq, VT *&alphaden, VT *&alpha,
                    VT *__restrict__ &u, VT *__restrict__ &res, VT *&nextResSq,
                    VT *&beta, gcxx::StreamView str1l, gcxx::StreamView str1h,
                    gcxx::StreamView str2, gcxx::StreamView str3,
                    ncclcommview ncomm) {

  nvtx3::scoped_range range{"main_loop"};
  // compute A * p (stream 1)
  nvtxRangePushA("Ap");
  launch_apply_stencil(str1l, str1h, numBlocks, blockSize, p, ap, nx, ny);
  nvtxRangePop();

  // alpha calculation depends on Ap (stream 1)
  nvtxRangePushA("alpha");
  launch_alphacalc(p, ap, nx, ny, numBlocks, blockSize, curResSq, alphaden,
                   alpha, str1l, ncomm);
  nvtxRangePop();

  // update solution (stream 2 - depends on alpha)
  nvtxRangePushA("solution");
  str2.WaitOnEvent(str1l.RecordEvent(gcxx::flags::eventCreate::disableTiming));
  gcxx::launch::Kernel(str2, numBlocks, blockSize, 0, cgUpdateSol<VT>, p, u,
                       alpha, nx, ny);
  nvtxRangePop();

  // update residual (stream 3 - depends on alpha, parallel with solution update)
  nvtxRangePushA("residual");
  str3.WaitOnEvent(str1l.RecordEvent(gcxx::flags::eventCreate::disableTiming));
  gcxx::launch::Kernel(str3, numBlocks, blockSize, 0, cgUpdateRes<VT>, ap, res,
                       alpha, nx, ny);
  nvtxRangePop();

  // compute residual norm (stream 3 - depends on residual update)
  nvtxRangePushA("resNorm");
  launch_resnormsqcalc(res, nx, ny, numBlocks, blockSize, nextResSq, str3,
                       ncomm);
  nvtxRangePop();

  // compute beta (stream 1 - depends on residual norm)
  nvtxRangePushA("beta");
  str1l.WaitOnEvent(str3.RecordEvent(gcxx::flags::eventCreate::disableTiming));
  gcxx::launch::Kernel(str1l, 1, 1, 0, gpu_devide<VT>, nextResSq, curResSq,
                       beta);
  gcxx::memory::Copy(curResSq, nextResSq, 1, str1l);
  nvtxRangePop();

  // update p (stream 1 - depends on beta)
  nvtxRangePushA("p");
  gcxx::launch::Kernel(str1l, numBlocks, blockSize, 0, cgUpdateP<VT>, beta, res,
                       p, nx, ny);
  nvtxRangePop();

  // Synchronize stream 1 to ensure all dependencies are met for next iteration
  str1l.WaitOnEvent(str2.RecordEvent(gcxx::flags::eventCreate::disableTiming));
}

template <typename VT>
inline size_t conjugateGradient(const VT *const __restrict__ rhs,
                                VT *__restrict__ u, VT *__restrict__ res,
                                VT *__restrict__ p, VT *__restrict__ ap,
                                const size_t nx, const size_t ny,
                                const size_t maxIt, const size_t ngraphsteps,
                                ncclcommview ncomm, int local_rank,
                                int local_size) {

  dim3 blockSize(blockSize_x, blockSize_y);
  int smcount = gcxx::Device::getAttribute(
    gcxx::flags::deviceAttribute::MultiProcessorCount);
  dim3 numBlocks(smcount, 10);

  gcxx::launch::Kernel(numBlocks, blockSize, residual_initp<VT>, res, p, rhs, u,
                       nx, ny);

  auto nextResSq_raii = nccl::make_nccl_unique<VT>(1);

  auto curResSq_raii = nccl::make_nccl_unique<VT>(1);
  auto alpha_raii = nccl::make_nccl_unique<VT>(1);
  auto beta_raii = nccl::make_nccl_unique<VT>(1);
  gcxx::memory::Memset(curResSq_raii, 0, 1);
  gcxx::memory::Memset(nextResSq_raii, 0, 1);
  gcxx::memory::Memset(alpha_raii, 0, 1);
  gcxx::memory::Memset(beta_raii, 0, 1);

  auto alphaden_raii = gcxx::memory::make_device_unique_ptr<VT>(1);
  VT *alphaden = alphaden_raii.get();

  // Create multiple streams for parallelization
  // Need to make them before the capture since streamdestroy is an synchronizing operation
  gcxx::Stream str1l(gcxx::flags::streamType::SyncWithNull,
                     gcxx::flags::streamPriority::VeryLow);
  gcxx::Stream str1h(gcxx::flags::streamType::SyncWithNull,
                     gcxx::flags::streamPriority::Critical);
  gcxx::Stream str2(gcxx::flags::streamType::SyncWithNull,
                    gcxx::flags::streamPriority::VeryLow);
  gcxx::Stream str3(gcxx::flags::streamType::SyncWithNull,
                    gcxx::flags::streamPriority::VeryLow);

  VT *curResSq = curResSq_raii.get();
  VT *nextResSq = nextResSq_raii.get();
  VT *alpha = alpha_raii.get();
  VT *beta = beta_raii.get();


  // compute residual norm
  launch_resnormsqcalc(res, nx, ny, numBlocks, blockSize, curResSq, str1l,
                       ncomm);


  bool graph_is_built = false;
  VT nextResSq_host{};
  gcxx::GraphExec graphexec;
  ssize_t ithost{0};
  do {
    if (!graph_is_built) {
      str1l.BeginCapture(gcxx::flags::streamCaptureMode::Global);
      for (int i = 0; i < ngraphsteps; ++i) {
        core_CG(numBlocks, blockSize, p, ap, nx, ny, curResSq, alphaden, alpha,
                u, res, nextResSq, beta, str1l, str1h, str2, str3, ncomm);
      }

      gcxx::memory::Copy(&nextResSq_host, nextResSq, 1, str1l);
      auto graph = str1l.EndCapture();
      graphexec = graph.Instantiate();
      graph_is_built = true;
    }
    graphexec.Launch(str1l);
    ++ithost;
    str1l.Synchronize();
    // if (local_rank == 0)
    //   fmt::print("maxiter {} | iter {}| res {}\n", maxIt, ngraphsteps * ithost,
    //              sqrt(nextResSq_host));

  } while (ngraphsteps * ithost < maxIt && sqrt(nextResSq_host) > 1e-12);

  return ithost * ngraphsteps;
}
