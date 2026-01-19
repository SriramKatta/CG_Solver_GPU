#pragma once

#include <mpi.h>

#ifndef SKIP_CUDA_AWARENESS_CHECK
#include <mpi-ext.h>
#if !defined(MPIX_CUDA_AWARE_SUPPORT) || !MPIX_CUDA_AWARE_SUPPORT
#error \
  "The used MPI Implementation does not have CUDA-aware support or CUDA-aware \
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

template <typename VT>
constexpr bool is_always_false_v = false;

template <typename VT>
struct mpitype {
  static_assert(is_always_false_v<VT>, "Unsupported MPI type");
};

template <>
struct mpitype<float> {
  static MPI_Datatype value() { return MPI_FLOAT; }
};

template <>
struct mpitype<double> {
  static MPI_Datatype value() { return MPI_DOUBLE; }
};

class request {
 private:
  MPI_Request req_{MPI_REQUEST_NULL};

 public:
  request() = default;
  request(request &&other) : req_(other.req_) { other.req_ = MPI_REQUEST_NULL; }
  request &operator=(request &&other) {
    req_ = other.req_;
    other.req_ = MPI_REQUEST_NULL;
    return *this;
  }
  MPI_Request &get() { return req_; }
  request(MPI_Request req) : req_(req) {}
  void wait() { MPI_Wait(&req_, MPI_STATUS_IGNORE); }
  ~request() { wait(); }
};


class mpicommview {
 protected:
  MPI_Comm comm_{MPI_COMM_NULL};

  mpicommview() = default;


 public:
  explicit mpicommview(MPI_Comm comm) : comm_(comm) {}
  MPI_Comm get() const { return comm_; }

  static mpicommview world() { return mpicommview{MPI_COMM_WORLD}; }

  int rank() const {
    int r{};
    MPI_CALL(MPI_Comm_rank(comm_, &r));
    return r;
  }

  int size() const {
    int s{};
    MPI_CALL(MPI_Comm_size(comm_, &s));
    return s;
  }

  void barrier() const { MPI_CALL(MPI_Barrier(comm_)); }

  void byte_bcast(void *value, size_t numbytes, int root) const {
    if (numbytes > std::numeric_limits<int>::max()) {
      throw std::runtime_error("MPI_Bcast count overflow");
    }
    MPI_CALL(
      MPI_Bcast(value, static_cast<int>(numbytes), MPI_BYTE, root, comm_));
  }

  template <typename VT>
  void reduce_sum(VT &val, int root) {
    if (rank() == root) {
      MPI_CALL(MPI_Reduce(MPI_IN_PLACE, &val, 1, mpitype<VT>::value(), MPI_SUM,
                          root, comm_));
    } else {
      MPI_CALL(MPI_Reduce(&val, nullptr, 1, mpitype<VT>::value(), MPI_SUM, root,
                          comm_));
    }
  }

  template <typename VT>
  void sendrecv(const VT *sendbuff, VT *recvbuff, size_t count, int sender_rank,
                int reciver_rank, int sender_tag, int reciver_tag) {
    auto type = mpitype<VT>::value();
    MPI_CALL(MPI_Sendrecv(sendbuff, count, type, reciver_rank, sender_tag,
                          recvbuff, count, type, sender_rank, reciver_tag,
                          comm_, MPI_STATUS_IGNORE));
  }

  template <typename VT>
  request isend(const VT *sendbuff, size_t count, int send_to_rank,
                    int sender_tag) {
    auto type = mpitype<VT>::value();
    MPI_Request req;
    MPI_CALL(
      MPI_Isend(sendbuff, count, type, send_to_rank, sender_tag, comm_, &req));
    return {req};
  }
  template <typename VT>
  request irecv(VT *recvbuff, size_t count, int recv_from_rank,
                    int recver_tag) {
    auto type = mpitype<VT>::value();
    MPI_Request req;
    MPI_CALL(MPI_Irecv(recvbuff, count, type, recv_from_rank, recver_tag, comm_,
                       &req));
    return {req};
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
