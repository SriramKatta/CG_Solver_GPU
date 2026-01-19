#pragma once

#include <fmt/format.h>
#include <cstring>
#include <string_view>
#include <tuple>
#include <vector>
#include "argparse/argparse.hpp"

#include "mpi_comm.hpp"
#include "util.h"

constexpr int app = 0;

template <typename tpe>
void initConjugateGradient(tpe *__restrict__ u, tpe *__restrict__ rhs,
                           const size_t nx, const size_t ny) {

  switch (app) {
    case 0:
      // version 0 - zero boundary, unit inner values
      for (size_t i1 = 0; i1 < ny; ++i1) {
        for (size_t i0 = 0; i0 < nx; ++i0) {
          if (0 == i0 || 0 == i1 || nx - 1 == i0 || ny - 1 == i1)
            u[i1 * nx + i0] = 0;
          else
            u[i1 * nx + i0] = 1;

          rhs[i1 * nx + i0] = 0;
        }
      }

      break;

    case 1:
      // version 1 - simple trigonometric test problem
      for (size_t i1 = 0; i1 < ny; ++i1) {
        for (size_t i0 = 0; i0 < nx; ++i0) {
          auto x = i0 - 1 + 0.5;
          auto y = i1 - 1 + 0.5;

          if (0 == i0 || 0 == i1 || nx - 1 == i0 || ny - 1 == i1)
            u[i1 * nx + i0] = sin(1e-3 * x) * cos(2e-3 * y);
          else
            u[i1 * nx + i0] = 1;
          rhs[i1 * nx + i0] = 5e-6 * sin(1e-3 * x) * cos(2e-3 * y);
        }
      }
      break;
  }
}

template <typename tpe>
void initConjugateGradientDistributed(tpe *__restrict__ u,
                                      tpe *__restrict__ rhs, size_t nx,
                                      size_t ny, mpicommview comm,
                                      int halo = 1) {
  int rank = comm.rank();
  int size = comm.size();

  if (size == 1) {
    return initConjugateGradient(u, rhs, nx, ny);
  }

  auto local_ny_with_halo = chunk_rows_of_rank(rank, size, ny);
  auto local_ny = rows_in_rank(rank, size, ny);  // actual rows without halos

  const bool is_top = is_top_check(rank, size);
  const bool is_bottom = is_bottom_check(rank, size);
  const bool is_central = is_central_check(rank, size);

  tpe *u_local = u;
  tpe *rhs_local = rhs;

  // needed to skip to prevent filling the halo elems
  if (!is_bottom) {
    u_local += nx;
    rhs_local += nx;
  }


  auto y_start = rows_start_of_rank(rank, size, ny);


  switch (app) {
    case 0: {
      for (size_t i1_local = 0; i1_local < local_ny; i1_local++) {
        for (size_t i0 = 0; i0 < nx; ++i0) {
          size_t i1 = i1_local + y_start;
          if (0 == i0 || 0 == i1 || nx - 1 == i0 || ny - 1 == i1)
            u_local[i1_local * nx + i0] = 0;
          else
            u_local[i1_local * nx + i0] = 1;

          rhs_local[i1_local * nx + i0] = 0;
        }
      }

    } break;
    case 1: {
      for (size_t i1_local = 0; i1_local < local_ny; i1_local++) {
        for (size_t i0 = 0; i0 < nx; ++i0) {
          size_t i1 = i1_local + y_start;
          auto x = i0 - 1 + 0.5;
          auto y = i1 - 1 + 0.5;

          if (0 == i0 || 0 == i1 || nx - 1 == i0 || ny - 1 == i1)
            u_local[i1_local * nx + i0] = sin(1e-3 * x) * cos(2e-3 * y);
          else
            u_local[i1_local * nx + i0] = 1;
          rhs_local[i1_local * nx + i0] = 5e-6 * sin(1e-3 * x) * cos(2e-3 * y);
        }
      }
    } break;
  }


  auto below_rank = get_rank_below(rank, size);
  auto above_rank = get_rank_above(rank, size);

  std::vector<request> reqs;

  if (is_top) {
    reqs.push_back(comm.isend(u_local, nx, below_rank, 0));
    reqs.push_back(comm.irecv(u, nx, below_rank, 0));

    reqs.push_back(comm.isend(rhs_local, nx, below_rank, 0));
    reqs.push_back(comm.irecv(rhs, nx, below_rank, 0));
  } else if (is_bottom) {
    reqs.push_back(
      comm.isend(u_local + (local_ny - 1) * nx, nx, above_rank, 0));
    reqs.push_back(comm.irecv(u_local + local_ny * nx, nx, above_rank, 0));

    reqs.push_back(
      comm.isend(rhs_local + (local_ny - 1) * nx, nx, above_rank, 0));
    reqs.push_back(comm.irecv(rhs_local + local_ny * nx, nx, above_rank, 0));
  } else if (is_central) {
    reqs.push_back(comm.isend(u_local, nx, below_rank, 0));
    reqs.push_back(
      comm.isend(u_local + (local_ny - 1) * nx, nx, above_rank, 0));
    reqs.push_back(comm.irecv(u, nx, below_rank, 0));
    reqs.push_back(comm.irecv(u_local + local_ny * nx, nx, above_rank, 0));


    reqs.push_back(comm.isend(rhs_local, nx, below_rank, 0));
    reqs.push_back(
      comm.isend(rhs_local + (local_ny - 1) * nx, nx, above_rank, 0));
    reqs.push_back(comm.irecv(rhs, nx, below_rank, 0));
    reqs.push_back(comm.irecv(rhs_local + local_ny * nx, nx, above_rank, 0));
  } else {
    fmt::print("something horrible is happening in init CG exiting program\n");
    std::exit(EXIT_FAILURE);
  }

  // Wait for all asynchronous communication to complete
  // MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
}

template <typename tpe>
void checkSolutionConjugateGradient(const tpe *const __restrict__ u,
                                    const tpe *const __restrict__ rhs,
                                    const size_t nx, const size_t ny) {
  double res = 0;
  for (size_t i1 = 1; i1 < ny - 1; ++i1) {
    for (size_t i0 = 1; i0 < nx - 1; ++i0) {
      const double localRes =
        rhs[i1 * nx + i0] -
        (4 * u[i1 * nx + i0] - (u[i1 * nx + i0 - 1] + u[i1 * nx + i0 + 1] +
                                u[(i1 - 1) * nx + i0] + u[(i1 + 1) * nx + i0]));

      res += localRes * localRes;
    }
  }

  res = sqrt(res);
  fmt::print("  Final residual is {}", res);
}

template <typename tpe>
void checkSolutionConjugateGradientDistributed(tpe *__restrict__ u,
                                               tpe *__restrict__ rhs, size_t nx,
                                               size_t ny) {
  double res = 0;
  for (size_t i1 = 1; i1 < ny - 1; ++i1) {
    for (size_t i0 = 1; i0 < nx - 1; ++i0) {
      const double localRes =
        rhs[i1 * nx + i0] -
        (4 * u[i1 * nx + i0] - (u[i1 * nx + i0 - 1] + u[i1 * nx + i0 + 1] +
                                u[(i1 - 1) * nx + i0] + u[(i1 + 1) * nx + i0]));

      res += localRes * localRes;
    }
  }

  auto world_comm = mpicomm::world();

  world_comm.reduce_sum(res, 0);


  res = sqrt(res);

  if (world_comm.rank() == 0)
    fmt::print("  Final residual is {}", res);
}

inline std::tuple<std::string, size_t, size_t, size_t, size_t, int> parseCLA_2d(
  int argc, char **argv) {
  argparse::ArgumentParser program("cg_solver");

  program.add_argument("tpeName")
    .help("type name for the simulation")
    .default_value("double")
    .required();

  program.add_argument("-x", "--nx")
    .help("grid size in x dimension")
    .default_value(size_t{4096})
    .scan<'u', size_t>();

  program.add_argument("-y", "--ny")
    .help("grid size in y dimension")
    .default_value(size_t{4096})
    .scan<'u', size_t>();

  program.add_argument("-n", "--nIt")
    .help("number of iterations")
    .default_value(size_t{20000})
    .scan<'u', size_t>();

  program.add_argument("-s", "--ngSteps")
    .help("number of iterations")
    .default_value(size_t{1})
    .scan<'u', size_t>();

  program.add_argument("-v", "--verbose")
    .help("increase verbosity level")
    .default_value(size_t{0})
    .scan<'u', size_t>();


  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    MPI_Finalize();
    std::exit(1);
  }


  auto tpeName = program.get<std::string>("tpeName");
  auto nx = program.get<size_t>("--nx");
  auto ny = program.get<size_t>("--ny");
  auto nIt = program.get<size_t>("--nIt");
  auto ngraphsteps = program.get<size_t>("--ngSteps");
  auto verbose = program.get<size_t>("--verbose");


  // fmt::print(
  //   "tpename {}\n"
  //   "nx {}\n"
  //   "ny {}\n"
  //   "nIt {}\n"
  //   "ngraphsteps {}\n"
  //   "verbose {}\n",
  //   tpeName, nx, ny, nIt, ngraphsteps, verbose

  //   );

    return {tpeName, nx, ny, nIt, ngraphsteps, verbose};
}