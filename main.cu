// V2 all gpu kernls only data moved once in each diection
#include "cg-util.h"
#include "kernel.cuh"
#include "mpi_comm.hpp"
#include "nccl_comm.hpp"

#include <fmt/format.h>
#include <string_view>

#include <gcxx/api.hpp>

#define NCCL_USE 1

template <typename VT>
inline int realMain(int argc, char *argv[]) {

  auto world_comm = mpicomm::world();
  int world_rank = world_comm.rank();
  int world_size = world_comm.size();

  auto devcount = gcxx::Device::count();

  if (world_size != devcount) {
    fmt::print("we need the device count to be equal to number of ranks");
    std::exit(EXIT_FAILURE);
  }

  int localrank = -1;
  int localsize = 0;
  {
    auto localcomm =
      mpicomm::split_type(world_comm, MPI_COMM_TYPE_SHARED, world_rank);
    localrank = localcomm.rank();
    localsize = localcomm.size();
  }

  ncclUniqueId ncclid{};
  if (world_rank == 0) {
    ncclid = nccl::getuniqueID();
  }

  world_comm.byte_bcast(&ncclid, sizeof(ncclid), 0);

  world_comm.barrier();
  auto devref = gcxx::Device::set(localrank);

  ncclcomm ncomm(localrank, localsize, ncclid);

  {  // scope may not be needed
    auto [tpeName, nx, ny, nIt] = parseCLA_2d(argc, argv);


    auto total_elems = nx * ny;

    auto u_host_raii =
      gcxx::memory::make_host_pinned_unique_ptr<VT>(total_elems);
    auto rhs_host_raii =
      gcxx::memory::make_host_pinned_unique_ptr<VT>(total_elems);

    VT *u_host = u_host_raii.get();
    VT *rhs_host = rhs_host_raii.get();

    // init
    initConjugateGradient(u_host, rhs_host, nx, ny);

#if NCCL_USE
    auto u_raii = nccl::make_nccl_unique<VT>(total_elems);
    auto rhs_raii = nccl::make_nccl_unique<VT>(total_elems);
#else
    auto u_raii = gcxx::memory::make_device_unique_ptr<VT>(total_elems);
    auto rhs_raii = gcxx::memory::make_device_unique_ptr<VT>(total_elems);
#endif

    VT *u = u_raii.get();
    VT *rhs = rhs_raii.get();

    gcxx::memory::Copy(u, u_host, total_elems);
    gcxx::memory::Copy(rhs, rhs_host, total_elems);


#if NCCL_USE
    auto res_raii = nccl::make_nccl_unique<VT>(total_elems);
    auto p_raii = nccl::make_nccl_unique<VT>(total_elems);
    auto ap_raii = nccl::make_nccl_unique<VT>(total_elems);
#else
    auto res_raii = gcxx::memory::make_device_unique_ptr<VT>(total_elems);
    auto p_raii = gcxx::memory::make_device_unique_ptr<VT>(total_elems);
    auto ap_raii = gcxx::memory::make_device_unique_ptr<VT>(total_elems);
#endif

    gcxx::memory::Memset(res_raii, 0, total_elems);
    gcxx::memory::Memset(p_raii, 0, total_elems);
    gcxx::memory::Memset(ap_raii, 0, total_elems);
    VT *res = res_raii.get();
    VT *p = p_raii.get();
    VT *ap = ap_raii.get();

    // warm-up
    // nItWarmUp = conjugateGradient(rhs, u, res, p, ap, nx, ny, nItWarmUp);

    // measurement
    auto start = std::chrono::steady_clock::now();

    nIt = conjugateGradient(rhs, u, res, p, ap, nx, ny, nIt);

    auto end = std::chrono::steady_clock::now();
    fmt::print("  CG steps:      {}\n", nIt);

    printStats<VT>(end - start, nIt, nx * ny, tpeName, 8 * sizeof(VT), 15);

    gcxx::memory::Copy(u_host, u, total_elems);
    gcxx::memory::Copy(rhs_host, rhs, total_elems);

    // check solution
    checkSolutionConjugateGradient(u_host, rhs_host, nx, ny);
  }  // scope may not be needed

  return 0;
}

int main(int argc, char *argv[]) {
  mpienv env(argc, argv);

  std::string_view tpeName;

  if (argc < 2) {
    fmt::print("Missing type specification using double");
    return -1;
  }

  tpeName = argv[1];

  if ("float" == tpeName)
    return realMain<float>(argc, argv);
  if ("double" == tpeName)
    return realMain<double>(argc, argv);

  fmt::print("Invalid type specification ({})\n\tfloat, double", tpeName);
  return -1;
}
