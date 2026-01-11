// V2 all gpu kernls only data moved once in each diection
#include "cg-util.h"

#include "cuda-util.h"

#include "kernel.cuh"

#include <fmt/format.h>
#include <string_view>

#include <gcxx/api.hpp>

template <typename VT>
inline int realMain(int argc, char *argv[]) {
  auto [tpeName, nx, ny, nItWarmUp, nIt] = parseCLA_2d(argc, argv);

  auto u_host_raii = gcxx::memory::make_host_pinned_unique_ptr<VT>(nx * ny);
  auto rhs_host_raii = gcxx::memory::make_host_pinned_unique_ptr<VT>(nx * ny);

  VT *u_host = u_host_raii.get();
  VT *rhs_host = rhs_host_raii.get();

  // init
  initConjugateGradient(u_host, rhs_host, nx, ny);

  auto u_raii = gcxx::memory::make_device_unique_ptr<VT>(nx * ny);
  auto rhs_raii = gcxx::memory::make_device_unique_ptr<VT>(nx * ny);

  VT *u = u_raii.get();
  VT *rhs = rhs_raii.get();

  gcxx::memory::Copy(u, u_host, nx * ny);
  gcxx::memory::Copy(rhs, rhs_host, nx * ny);


  auto res_raii = gcxx::memory::make_device_unique_ptr<VT>(nx * ny);
  auto p_raii = gcxx::memory::make_device_unique_ptr<VT>(nx * ny);
  auto ap_raii = gcxx::memory::make_device_unique_ptr<VT>(nx * ny);
  gcxx::memory::Memset(res_raii, 0, nx * ny);
  gcxx::memory::Memset(p_raii, 0, nx * ny);
  gcxx::memory::Memset(ap_raii, 0, nx * ny);
  VT *res = res_raii.get();
  VT *p = p_raii.get();
  VT *ap = ap_raii.get();

  // warm-up
  nItWarmUp = conjugateGradient(rhs, u, res, p, ap, nx, ny, nItWarmUp);

  // measurement
  auto start = std::chrono::steady_clock::now();

  nIt = conjugateGradient(rhs, u, res, p, ap, nx, ny, nIt);

  auto end = std::chrono::steady_clock::now();
  fmt::print("  CG steps:      {}\n", nIt);

  printStats<VT>(end - start, nIt, nx * ny, tpeName, 8 * sizeof(VT), 15);

  gcxx::memory::Copy(u_host, u, nx*ny);
  gcxx::memory::Copy(rhs_host, rhs, nx*ny);

  // check solution
  checkSolutionConjugateGradient(u_host, rhs_host, nx, ny);


  return 0;
}

int main(int argc, char *argv[]) {

  std::string_view tpeName;

  if (argc < 2) {
    fmt::print("Missing type specification using double");
    tpeName = "double";
  }

  tpeName = argv[1];

  if ("float" == tpeName)
    return realMain<float>(argc, argv);
  if ("double" == tpeName)
    return realMain<double>(argc, argv);

  fmt::print("Invalid type specification ({})\n\tfloat, double", tpeName);
  return -1;
}
