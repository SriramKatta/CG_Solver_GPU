#pragma once

#include <mpi.h>

#ifndef SKIP_CUDA_AWARENESS_CHECK
#include <mpi-ext.h>
#if !defined(MPIX_CUDA_AWARE_SUPPORT) || !MPIX_CUDA_AWARE_SUPPORT
#error "The used MPI Implementation does not have CUDA-aware support or CUDA-aware \
support can't be determined. Define SKIP_CUDA_AWARENESS_CHECK to skip this check."
#endif
#endif

#define MPI_CALL(call)                                                    \
  {                                                                       \
    int mpi_status = call;                                                \
    if (MPI_SUCCESS != mpi_status) {                                      \
      char mpi_error_string[MPI_MAX_ERROR_STRING];                        \
      int mpi_error_string_length = 0;                                    \
      MPI_Error_string(mpi_status, mpi_error_string,                      \
                       &mpi_error_string_length);                         \
      if (NULL != mpi_error_string)                                       \
        fprintf(stderr,                                                   \
                "ERROR: MPI call \"%s\" in line %d of file %s failed "    \
                "with %s "                                                \
                "(%d).\n",                                                \
                #call, __LINE__, __FILE__, mpi_error_string, mpi_status); \
      else                                                                \
        fprintf(stderr,                                                   \
                "ERROR: MPI call \"%s\" in line %d of file %s failed "    \
                "with %d.\n",                                             \
                #call, __LINE__, __FILE__, mpi_status);                   \
      exit(mpi_status);                                                   \
    }                                                                     \
  }

struct mpienv {
  mpienv(int &argc, char **&argv) { MPI_CALL(MPI_Init(&argc, &argv)); }
  ~mpienv() { MPI_CALL(MPI_Finalize()); }
};

class mpicommview {
 protected:
  MPI_Comm comm_{MPI_COMM_NULL};
  mpicommview(MPI_Comm comm) : comm_(comm) {}

 public:
  mpicommview() = default;
  MPI_Comm get() const { return comm_; }
  static mpicommview world() { return {MPI_COMM_WORLD}; }
  int rank() const {
    int r{};
    MPI_CALL(MPI_Comm_rank(comm_, &r));
    return r;
  }

  int size() const {
    int r{};
    MPI_CALL(MPI_Comm_size(comm_, &r));
    return r;
  }

  void barrier() const { MPI_CALL(MPI_Barrier(comm_)); }

  void byte_bcast(void *value, size_t numbytes, int root) {
    MPI_CALL(MPI_Bcast(value, numbytes, MPI_BYTE, root, comm_));
  }
};

class mpicomm : public mpicommview {
  mpicomm(MPI_Comm comm) : mpicommview(comm) {}

 public:
  static mpicomm split(const mpicommview &basecomm, int color, int key) {
    MPI_Comm loccomm;
    MPI_CALL(MPI_Comm_split(basecomm.get(), color, key, &loccomm))
    return {loccomm};
  }

  static mpicomm split_type(const mpicommview &basecomm, int type, int key) {
    MPI_Comm loccomm;
    MPI_CALL(
      MPI_Comm_split_type(basecomm.get(), type, key, MPI_INFO_NULL, &loccomm))
    return {loccomm};
  }

  mpicomm(const mpicommview &basecomm, const MPI_Group &grp) : mpicommview() {
    MPI_CALL(MPI_Comm_create(basecomm.get(), grp, &comm_));
  }
  ~mpicomm() {
    if (comm_ != MPI_COMM_NULL)
      MPI_CALL(MPI_Comm_free(&comm_));
  }
};
