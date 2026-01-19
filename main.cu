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
inline int realMain(std::string_view tpeName, size_t nx, size_t ny_global,
                    size_t nIt) {

  auto world_comm = mpicomm::world();
  int world_rank = world_comm.rank();
  int world_size = world_comm.size();


  auto devcount = gcxx::Device::count();


  if (world_size > devcount) {
    fmt::print("we don not support multiple ranks per GPU");
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


  // fmt::print("wrank {}, wsize {}, gpu count {}\n", world_rank, world_size,
  //            devcount);
  // fmt::print("local rank {} | local size {}\n", localrank, localsize);

  // return 0;

  ncclUniqueId ncclid{};
  if (world_rank == 0) {
    ncclid = nccl::getuniqueID();
  }

  world_comm.byte_bcast(&ncclid, sizeof(ncclid), 0);

  world_comm.barrier();
  auto devref = gcxx::Device::set(localrank);


  ncclcomm ncomm(localrank, localsize, ncclid);

  auto chunk_size_with_halo =
    chunk_rows_of_rank(world_rank, world_size, ny_global);

  auto total_elems = nx * chunk_size_with_halo;

  auto u_host_raii = gcxx::memory::make_host_pinned_unique_ptr<VT>(total_elems);
  auto rhs_host_raii =
    gcxx::memory::make_host_pinned_unique_ptr<VT>(total_elems);

  VT *u_host = u_host_raii.get();
  VT *rhs_host = rhs_host_raii.get();

  // init
  // initConjugateGradient(u_host, rhs_host, nx, ny);
  initConjugateGradientDistributed(
    u_host, rhs_host, nx, ny_global,
    world_comm);  // TODO think and change to local comm

  auto u_raii = gcxx::memory::make_device_unique_ptr<VT>(total_elems);
  auto rhs_raii = gcxx::memory::make_device_unique_ptr<VT>(total_elems);

  VT *u = u_raii.get();
  VT *rhs = rhs_raii.get();


  gcxx::memory::Copy(u, u_host, total_elems);
  gcxx::memory::Copy(rhs, rhs_host, total_elems);

  auto res_raii = gcxx::memory::make_device_unique_ptr<VT>(total_elems);
  auto p_raii = gcxx::memory::make_device_unique_ptr<VT>(total_elems);
  auto ap_raii = gcxx::memory::make_device_unique_ptr<VT>(total_elems);


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

  nIt = conjugateGradient(rhs, u, res, p, ap, nx, chunk_size_with_halo, nIt,
                          ncomm, localrank, localsize);

  auto end = std::chrono::steady_clock::now();

  gcxx::memory::Copy(u_host, u, total_elems);
  gcxx::memory::Copy(rhs_host, rhs, total_elems);

  // check solution
  if (world_rank == 0) {
    fmt::print("  CG steps:      {}\n", nIt);
    printStats<VT>(end - start, nIt, nx * ny_global, tpeName, 8 * sizeof(VT),
                   15);
  }

  checkSolutionConjugateGradientDistributed(u_host, rhs_host, nx,
                                            chunk_size_with_halo);

  return 0;
}

int main(int argc, char *argv[]) try {
  mpienv env(argc, argv);

  auto [tpeName, nx, ny_global, nIt] = parseCLA_2d(argc, argv);
  if ("float" == tpeName)
    return realMain<float>(tpeName, nx, ny_global, nIt);
  if ("double" == tpeName)
    return realMain<double>(tpeName, nx, ny_global, nIt);

} catch (...) {
  return EXIT_FAILURE;
}
