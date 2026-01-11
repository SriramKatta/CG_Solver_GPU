// V2 all gpu kernls only data moved once in each diection
#include "cg-util.h"

#include "cuda-util.h"

#include <cub/block/block_reduce.cuh>

#include <nvtx3/nvtx3.hpp>


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

  if (threadIdx.x == 0) {
    *res = (*num) / (*den);
  }
}

template <typename VT>
void resnormsqcalc(const VT *const __restrict__ res, size_t nx, size_t ny,
                   dim3 numblocks, dim3 blocksize, VT *ressqnorm) {
  size_t smemsize = blocksize.x * blocksize.y * sizeof(VT);
  cudaMemset(ressqnorm, 0, sizeof(VT));
  innerproduct<<<numblocks, blocksize, smemsize>>>(res, res, ressqnorm, nx, ny);
}

template <typename VT>
void alphacalc(const VT *const __restrict__ p, const VT *const __restrict__ ap,
               size_t nx, size_t ny, dim3 numblocks, dim3 blocksize,
               VT *alphanum, VT *alpha) {
  size_t smemsize = blocksize.x * blocksize.y * sizeof(VT);
  VT *alphaden;
  VT hostalphaden = 1.0;
  cudaMalloc(&alphaden, sizeof(VT));
  cudaMemset(alphaden, 0, sizeof(VT));
  innerproduct<<<numblocks, blocksize, smemsize>>>(p, ap, alphaden, nx, ny);
  gpu_devide<<<1, 1>>>(alphanum, alphaden, alpha);
}

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

  // initialization
  //VT initResSq = (VT)0;

  residual_initp<VT><<<numBlocks, blockSize>>>(res, p, rhs, u, nx, ny);

  VT *curResSq;
  cudaMallocManaged(&curResSq, sizeof(VT));
  cudaMemset(curResSq, 0, sizeof(VT));
  VT *nextResSq;
  cudaMallocManaged(&nextResSq, sizeof(VT));
  cudaMemset(nextResSq, 0, sizeof(VT));
  VT *alpha;
  cudaMallocManaged(&alpha, sizeof(VT));
  cudaMemset(alpha, 0, sizeof(VT));
  VT *beta;
  cudaMallocManaged(&beta, sizeof(VT));
  cudaMemset(beta, 0, sizeof(VT));

  // compute residual norm
  resnormsqcalc(res, nx, ny, numBlocks, blockSize, curResSq);

  // main loop
  for (size_t it = 0; it < maxIt; ++it) {
    nvtx3::scoped_range loop{"main loop"};

    // compute A * p
    nvtxRangePushA("Ap");
    applystencil<VT><<<numBlocks, blockSize>>>(p, ap, nx, ny);
    nvtxRangePop();

    nvtxRangePushA("alpha");
    alphacalc(p, ap, nx, ny, numBlocks, blockSize, curResSq, alpha);
    nvtxRangePop();

    // update solution
    nvtxRangePushA("solution");
    cgUpdateSol<VT><<<numBlocks, blockSize>>>(p, u, alpha, nx, ny);
    nvtxRangePop();

    // update residual
    nvtxRangePushA("residual");
    cgUpdateRes<VT><<<numBlocks, blockSize>>>(ap, res, alpha, nx, ny);
    nvtxRangePop();

    // compute residual norm
    nvtxRangePushA("resNorm");
    resnormsqcalc(res, nx, ny, numBlocks, blockSize, nextResSq);
    nvtxRangePop();

    // check exit criterion
    cudaMemPrefetchAsync(nextResSq, sizeof(VT), cudaCpuDeviceId);
    checkCudaError(cudaDeviceSynchronize()); // Add this!
    if (sqrt(*nextResSq) <= 1e-12) {
      return it;
    }

    if (0 == it % 100)
      std::cout << "    " << it << " : " << sqrt(*nextResSq) << std::endl;

    // compute beta
    cudaMemPrefetchAsync(nextResSq, sizeof(VT), cudaCpuDeviceId);
    nvtxRangePushA("beta");
    gpu_devide<<<1, 1>>>(nextResSq, curResSq, beta);
    cudaMemcpy(curResSq, nextResSq, sizeof(VT), cudaMemcpyDeviceToDevice);
    nvtxRangePop();

    // update p
    nvtxRangePushA("p");
    cgUpdateP<<<numBlocks, blockSize>>>(beta, res, p, nx, ny);
    // checkCudaError(cudaDeviceSynchronize());
    nvtxRangePop();
  }

  return maxIt;
}

template <typename VT>
inline int realMain(int argc, char *argv[]) {
  char *tpeName;
  size_t nx, ny, nItWarmUp, nIt;
  parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt);

  VT *u_host;
  VT *rhs_host;
  checkCudaError(cudaMallocHost(&u_host, sizeof(VT) * nx * ny));
  checkCudaError(cudaMallocHost(&rhs_host, sizeof(VT) * nx * ny));

  // init
  initConjugateGradient(u_host, rhs_host, nx, ny);

  VT *u;
  VT *rhs;
  checkCudaError(cudaMalloc(&u, sizeof(VT) * nx * ny));
  checkCudaError(cudaMalloc(&rhs, sizeof(VT) * nx * ny));
  checkCudaError(
    cudaMemcpy(u, u_host, sizeof(VT) * nx * ny, cudaMemcpyHostToDevice));
  checkCudaError(
    cudaMemcpy(rhs, rhs_host, sizeof(VT) * nx * ny, cudaMemcpyHostToDevice));

  VT *res;
  checkCudaError(cudaMalloc(&res, sizeof(VT) * nx * ny));
  VT *p;
  checkCudaError(cudaMalloc(&p, sizeof(VT) * nx * ny));
  VT *ap;
  checkCudaError(cudaMalloc(&ap, sizeof(VT) * nx * ny));

  checkCudaError(cudaMemset(res, 0, sizeof(VT) * nx * ny));
  checkCudaError(cudaMemset(p, 0, sizeof(VT) * nx * ny));
  checkCudaError(cudaMemset(ap, 0, sizeof(VT) * nx * ny));

  // warm-up
  nItWarmUp = conjugateGradient(rhs, u, res, p, ap, nx, ny, nItWarmUp);

  // measurement
  auto start = std::chrono::steady_clock::now();

  nIt = conjugateGradient(rhs, u, res, p, ap, nx, ny, nIt);

  auto end = std::chrono::steady_clock::now();
  std::cout << "  CG steps:      " << nIt << std::endl;

  printStats<VT>(end - start, nIt, nx * ny, tpeName, 8 * sizeof(VT), 15);

  checkCudaError(
    cudaMemcpy(u_host, u, sizeof(VT) * nx * ny, cudaMemcpyDeviceToHost));
  checkCudaError(
    cudaMemcpy(rhs_host, rhs, sizeof(VT) * nx * ny, cudaMemcpyDeviceToHost));

  // check solution
  checkSolutionConjugateGradient(u_host, rhs_host, nx, ny);

  checkCudaError(cudaFree(res));
  checkCudaError(cudaFree(p));
  checkCudaError(cudaFree(ap));

  checkCudaError(cudaFreeHost(u_host));
  checkCudaError(cudaFreeHost(rhs_host));

  checkCudaError(cudaFree(u));
  checkCudaError(cudaFree(rhs));

  return 0;
}

int main(int argc, char *argv[]) {
  //if (argc < 2) {
  //  std::cout << "Missing type specification " << std::endl;
  //  return 1;
  //}

  std::string tpeName(argv[1]);

  if ("float" == tpeName)
    return realMain<float>(argc, argv);
  if ("double" == tpeName || tpeName.empty())
    return realMain<double>(argc, argv);

  std::cout << "Invalid type specification (" << argv[1]
            << "); supported types are" << std::endl;
  std::cout << "  float, double" << std::endl;
  return -1;
}
