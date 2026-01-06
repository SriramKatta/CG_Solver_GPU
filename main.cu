// V2 all gpu kernls only data moved once in each diection
#include "cg-util.h"

#include <gcxx/api.hpp>

#include <nvtx3/nvtx3.hpp>

#include "kernels.cuh"

template <typename VT>
inline size_t conjugateGradient(const VT *const __restrict__ rhs,
                                VT *__restrict__ u, VT *__restrict__ res,
                                VT *__restrict__ p, VT *__restrict__ ap,
                                const size_t nx, const size_t ny,
                                const size_t maxIt) {

  constexpr auto blockSize_x = 32, blockSize_y = 16;
  dim3 blockSize(blockSize_x, blockSize_y);
  int smcount = gcxx::Device::getAttribute(gcxx::flags::deviceAttribute::MultiProcessorCount);
  dim3 numBlocks(smcount, 10);

  // initialization
  VT initResSq = (VT)0;

  residual_initp<VT><<<numBlocks, blockSize>>>(res, p, rhs, u, nx, ny);

  // compute residual norm
  VT curResSq = resnormsqcalc(res, nx, ny, numBlocks, blockSize);

  // main loop
  size_t it =0;
  while(true) {
    nvtx3::scoped_range loop{"main loop"};

    nvtxRangePushA("Ap");
    // compute A * p
    applystencil<VT><<<numBlocks, blockSize>>>(p, ap, nx, ny);
    gcxx::Device::Synchronize();
    nvtxRangePop();

    nvtxRangePushA("alpha");
    VT alphaNominator = curResSq;
    VT alphaDenominator = alphadencalc(p, ap, nx, ny, numBlocks, blockSize);
    VT alpha = alphaNominator / alphaDenominator;
    nvtxRangePop();

    // update solution
    nvtxRangePushA("solution");
    cgUpdateSol<VT><<<numBlocks, blockSize>>>(p, u, alpha, nx, ny);
    gcxx::Device::Synchronize();
    nvtxRangePop();

    // update residual
    nvtxRangePushA("residual");
    cgUpdateRes<VT><<<numBlocks, blockSize>>>(ap, res, alpha, nx, ny);
    gcxx::Device::Synchronize();
    nvtxRangePop();

    // compute residual norm
    nvtxRangePushA("resNorm");
    VT nextResSq = resnormsqcalc(res, nx, ny, numBlocks, blockSize);
    nvtxRangePop();

    // check exit criterion
    if (sqrt(nextResSq) <= 1e-12)
      return it;

    ++it;
    if (0 == it % 100)
      std::cout << "    " << it << " : " << sqrt(nextResSq) << std::endl;

    // compute beta
    nvtxRangePushA("beta");
    VT beta = nextResSq / curResSq;
    curResSq = nextResSq;
    nvtxRangePop();

    // update p
    nvtxRangePushA("p");
    cgUpdateP<<<numBlocks, blockSize>>>(beta, res, p, nx, ny);
    gcxx::Device::Synchronize();
    nvtxRangePop();
  }

  return maxIt;
}

template <typename VT>
inline int realMain(int argc, char *argv[]) {
  char *tpeName;
  size_t nx, ny, nItWarmUp, nIt;
  parseCLA_2d(argc, argv, tpeName, nx, ny, nItWarmUp, nIt);

  auto u_host = gcxx::host_vector<VT>(nx * ny);
  auto u_host_span = gcxx::span{u_host.data(), nx * ny};
  auto rhs_host = gcxx::host_vector<VT>(nx * ny);
  auto rhs_host_span = gcxx::span{rhs_host.data(), nx * ny};


  // init
  initConjugateGradient(u_host.data(), rhs_host.data(), nx, ny);

  auto u = gcxx::device_vector<VT>(nx * ny);
  auto u_span = gcxx::span{u.data(), nx * ny};
  auto rhs = gcxx::device_vector<VT>(nx * ny);
  auto rhs_span = gcxx::span{rhs.data(), nx * ny};


  gcxx::memory::Copy(u_span, u_host_span);
  gcxx::memory::Copy(rhs_span, rhs_host_span);

  auto res = gcxx::device_vector<VT>(nx * ny);
  auto res_span = gcxx::span(res.data(), nx * ny);
  auto p = gcxx::device_vector<VT>(nx * ny);
  auto p_span = gcxx::span(p.data(), nx * ny);
  auto ap = gcxx::device_vector<VT>(nx * ny);
  auto ap_span = gcxx::span(ap.data(), nx * ny);

  gcxx::memory::Memset(res_span, 0);
  gcxx::memory::Memset(p_span, 0);
  gcxx::memory::Memset(ap_span, 0);

  // warm-up
  // nItWarmUp = conjugateGradient(rhs.data(), u.data(), res.data(), p.data(),
  //                               ap.data(), nx, ny, nItWarmUp);

  // measurement
  auto start = std::chrono::steady_clock::now();

  nIt = conjugateGradient(rhs.data(), u.data(), res.data(), p.data(), ap.data(),
                          nx, ny, nItWarmUp);
  std::cout << "  CG steps:      " << nIt << std::endl;

  auto end = std::chrono::steady_clock::now();

  printStats<VT>(end - start, nIt, nx * ny, tpeName, 8 * sizeof(VT), 15);

  gcxx::memory::Copy(u_host_span, u_span);
  gcxx::memory::Copy(rhs_host_span, rhs_span);


  // check solution
  checkSolutionConjugateGradient(u_host.data(), rhs_host.data(), nx, ny);


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
