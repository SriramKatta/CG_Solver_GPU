// V1 added prefetching to all kernls to page fault data transfer between cpu and gpu
// V2 make everything grid stride loop
// V3 made dot product reduction kernel and done all the computation on gpu
#include "cg-util.h"

#include "cuda-util.h"

#include <nvtx3/nvtx3.hpp>

template <typename tpe>
__global__ void cgAp(const tpe *const __restrict__ p, tpe *__restrict__ ap,
                     const size_t nx, const size_t ny) {
  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY) {
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      ap[j * nx + i] =
        4 * p[j * nx + i] - (p[j * nx + i - 1] + p[j * nx + i + 1] +
                             p[(j - 1) * nx + i] + p[(j + 1) * nx + i]);
    }
  }
}

template <typename tpe>
__global__ void cgUpdateSol(const tpe *const __restrict__ p,
                            tpe *__restrict__ u, const tpe alpha,
                            const size_t nx, const size_t ny) {

  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY) {
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      u[j * nx + i] += alpha * p[j * nx + i];
    }
  }
}

template <typename tpe>
__global__ void cgUpdateRes(const tpe *const __restrict__ ap,
                            tpe *__restrict__ res, const tpe alpha,
                            const size_t nx, const size_t ny) {

  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY) {
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      res[j * nx + i] = res[j * nx + i] - alpha * ap[j * nx + i];
    }
  }
}

template <typename tpe>
__global__ void cgUpdateP(tpe beta, const tpe *const __restrict__ res,
                          tpe *__restrict__ p, size_t nx, size_t ny) {

  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY) {
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      p[j * nx + i] = res[j * nx + i] + beta * p[j * nx + i];
    }
  }
}

template <typename tpe>
__global__ void residual_initp(tpe *__restrict__ res, tpe *__restrict__ p,
                               const tpe *const __restrict__ rhs,
                               const tpe *const __restrict__ u, size_t nx,
                               size_t ny) {

  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;

  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY) {
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      auto temp =
        rhs[j * nx + i] -
        (4 * u[j * nx + i] - (u[j * nx + i - 1] + u[j * nx + i + 1] +
                              u[(j - 1) * nx + i] + u[(j + 1) * nx + i]));
      res[j * nx + i] = temp;
      p[j * nx + i] = temp;
    }
  }
}

template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
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

template <typename tpe>
__global__ void dotproduct(const tpe *const __restrict__ a,
                           const tpe *const __restrict__ b, tpe *result,
                           size_t nx, size_t ny) {

  size_t gridStartX = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t gridStrideX = gridDim.x * blockDim.x;
  size_t gridStartY = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t gridStrideY = gridDim.y * blockDim.y;
  tpe *smem = SharedMemory<tpe>();
  tpe sum = (tpe)0;
  for (size_t j = gridStartY; j < ny - 1; j += gridStrideY) {
    for (size_t i = gridStartX; i < nx - 1; i += gridStrideX) {
      const auto gindex = j * nx + i;
      sum += a[gindex] * b[gindex];
    }
  }
  auto tid = threadIdx.y * blockDim.x + threadIdx.x;
  smem[tid] = sum;
  for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
  }

  if (0 == tid) {
    atomicAdd(result, smem[0]);
  }
}

template <typename tpe>
tpe resudulanormcalc(const tpe *const __restrict__ res, const size_t &nx,
                     const size_t &ny, const dim3 &blocksize,
                     const dim3 &numblocks) {
  size_t smemesize = blocksize.x * blocksize.y * sizeof(tpe);
  tpe *resnormsq;
  cudaMallocManaged(&resnormsq, sizeof(tpe));
  cudaMemset(resnormsq, 0, sizeof(tpe));
  dotproduct<<<numblocks, blocksize, smemesize>>>(res, res, resnormsq, nx, ny);
  checkCudaError(cudaDeviceSynchronize());
  return *resnormsq;
}

template <typename tpe>
tpe alphaDenominatorcalc(const tpe *const __restrict__ p,
                         const tpe *const __restrict__ ap, const size_t &nx,
                         const size_t &ny, const dim3 &blocksize,
                         const dim3 &numblocks) {
  size_t smemesize = blocksize.x * blocksize.y * sizeof(tpe);
  tpe *alphaDenominator;
  cudaMallocManaged(&alphaDenominator, sizeof(tpe));
  cudaMemset(alphaDenominator, 0, sizeof(tpe));
  dotproduct<<<numblocks, blocksize, smemesize>>>(p, ap, alphaDenominator, nx,
                                                  ny);
  checkCudaError(cudaDeviceSynchronize());
  return *alphaDenominator;
}

template <typename tpe>
inline size_t conjugateGradient(const tpe *const __restrict__ rhs,
                                tpe *__restrict__ u, tpe *__restrict__ res,
                                tpe *__restrict__ p, tpe *__restrict__ ap,
                                const size_t nx, const size_t ny,
                                const size_t maxIt) {

  constexpr auto blockSize_x = 32, blockSize_y = 16;
  dim3 blockSize(blockSize_x, blockSize_y);
  int smcount = 0;
  cudaDeviceGetAttribute(&smcount, cudaDevAttrMultiProcessorCount, 0);
  dim3 numBlocks(smcount, 10);

  residual_initp<tpe><<<numBlocks, blockSize>>>(res, p, rhs, u, nx, ny);

  // compute residual norm
  tpe curResSq = resudulanormcalc(res, nx, ny, blockSize, numBlocks);

  // main loop
  for (size_t it = 0; it < maxIt; ++it) {
    nvtx3::scoped_range loop{"main loop"};

    nvtxRangePushA("Ap");
    // compute A * p
    cgAp<tpe><<<numBlocks, blockSize>>>(p, ap, nx, ny);
    checkCudaError(cudaDeviceSynchronize());
    nvtxRangePop();

    nvtxRangePushA("alpha");
    tpe alphaNominator = curResSq;
    tpe alphaDenominator =
      alphaDenominatorcalc(p, ap, nx, ny, blockSize, numBlocks);
    tpe alpha = alphaNominator / alphaDenominator;
    nvtxRangePop();

    // update solution
    nvtxRangePushA("solution");
    cgUpdateSol<tpe><<<numBlocks, blockSize>>>(p, u, alpha, nx, ny);
    checkCudaError(cudaDeviceSynchronize());
    nvtxRangePop();

    // update residual
    nvtxRangePushA("residual");
    cgUpdateRes<tpe><<<numBlocks, blockSize>>>(ap, res, alpha, nx, ny);
    checkCudaError(cudaDeviceSynchronize());
    nvtxRangePop();

    // compute residual norm
    nvtxRangePushA("resNorm");
    tpe nextResSq = resudulanormcalc(res, nx, ny, blockSize, numBlocks);
    nvtxRangePop();

    // check exit criterion
    if (sqrt(nextResSq) <= 1e-12)
      return it;

    // if (0 == it % 100)
    //     std::cout << "    " << it << " : " << sqrt(nextResSq) << std::endl;

    // compute beta
    nvtxRangePushA("beta");
    tpe beta = nextResSq / curResSq;
    curResSq = nextResSq;
    nvtxRangePop();

    // update p
    nvtxRangePushA("p");
    cgUpdateP<<<numBlocks, blockSize>>>(beta, res, p, nx, ny);
    checkCudaError(cudaDeviceSynchronize());
    nvtxRangePop();
  }

  return maxIt;
}

template <typename tpe>
inline int realMain(int argc, char *argv[]) {
  char *tpeName;
  size_t nx, ny, nItWarmUp, nIt;
  parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt);

  tpe *u;
  checkCudaError(cudaMallocManaged(&u, sizeof(tpe) * nx * ny));
  tpe *rhs;
  checkCudaError(cudaMallocManaged(&rhs, sizeof(tpe) * nx * ny));

  // init
  initConjugateGradient(u, rhs, nx, ny);

  checkCudaError(cudaMemPrefetchAsync(u, sizeof(tpe) * nx * ny, 0));
  checkCudaError(cudaMemPrefetchAsync(rhs, sizeof(tpe) * nx * ny, 0));

  tpe *res;
  checkCudaError(cudaMallocManaged(&res, sizeof(tpe) * nx * ny));
  tpe *p;
  checkCudaError(cudaMallocManaged(&p, sizeof(tpe) * nx * ny));
  tpe *ap;
  checkCudaError(cudaMallocManaged(&ap, sizeof(tpe) * nx * ny));

  checkCudaError(cudaMemset(res, 0, sizeof(tpe) * nx * ny));
  checkCudaError(cudaMemset(p, 0, sizeof(tpe) * nx * ny));
  checkCudaError(cudaMemset(ap, 0, sizeof(tpe) * nx * ny));

  // warm-up
  nItWarmUp = conjugateGradient(rhs, u, res, p, ap, nx, ny, nItWarmUp);

  // measurement
  auto start = std::chrono::steady_clock::now();

  nIt = conjugateGradient(rhs, u, res, p, ap, nx, ny, nIt);
  std::cout << "  CG steps:      " << nIt << std::endl;

  auto end = std::chrono::steady_clock::now();

  printStats<tpe>(end - start, nIt, nx * ny, tpeName, 8 * sizeof(tpe), 15);

  checkCudaError(
    cudaMemPrefetchAsync(u, sizeof(tpe) * nx * ny, cudaCpuDeviceId));
  checkCudaError(
    cudaMemPrefetchAsync(rhs, sizeof(tpe) * nx * ny, cudaCpuDeviceId));

  // check solution
  checkSolutionConjugateGradient(u, rhs, nx, ny);

  checkCudaError(cudaFree(res));
  checkCudaError(cudaFree(p));
  checkCudaError(cudaFree(ap));

  checkCudaError(cudaFree(u));
  checkCudaError(cudaFree(rhs));

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Missing type specification " << std::endl;
    return 1;
  }

  std::string tpeName(argv[1]);

  if ("float" == tpeName)
    return realMain<float>(argc, argv);
  if ("double" == tpeName)
    return realMain<double>(argc, argv);

  std::cout << "Invalid type specification (" << argv[1]
            << "); supported types are" << std::endl;
  std::cout << "  float, double" << std::endl;
  return -1;
}
