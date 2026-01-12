#pragma once

#include <nccl.h>
#include <gcxx/api.hpp>

#define NCCL_CALL(call)                                                  \
  {                                                                      \
    ncclResult_t ncclStatus = call;                                      \
    if (ncclSuccess != ncclStatus) {                                     \
      fprintf(stderr,                                                    \
              "ERROR: NCCL call \"%s\" in line %d of file %s failed "    \
              "with "                                                    \
              "%s (%d).\n",                                              \
              #call, __LINE__, __FILE__, ncclGetErrorString(ncclStatus), \
              ncclStatus);                                               \
      exit(ncclStatus);                                                  \
    }                                                                    \
  }

namespace nccl {
  ncclUniqueId getuniqueID() {
    ncclUniqueId nccl_uid;
    NCCL_CALL(ncclGetUniqueId(&nccl_uid));
    return nccl_uid;
  }
  void groupstart() {
    NCCL_CALL(ncclGroupStart());
  }

  void groupend() {
    NCCL_CALL(ncclGroupEnd());
  }

  auto NcclMemDeleter = [](void *ptr) {
    if (ptr) {
      NCCL_CALL(ncclMemFree(ptr));
    }
  };

  template <typename VT>
  inline std::unique_ptr<VT, decltype(NcclMemDeleter)> make_nccl_unique(
    size_t numelems) {
    void *ptr = nullptr;
    NCCL_CALL(ncclMemAlloc(&ptr, sizeof(VT) * numelems));
    return {static_cast<VT *>(ptr), NcclMemDeleter};
  }

}  // namespace nccl

struct reghandle {
  ncclComm_t comm_{NCCL_COMM_NULL};
  void *handle_{nullptr};

  reghandle() = default;

  reghandle(ncclComm_t comm, const void *ptr, size_t bytes) : comm_(comm) {
    NCCL_CALL(
      ncclCommRegister(comm_, const_cast<void *>(ptr), bytes, &handle_));
  }

  void destroy() {
    if (handle_ && comm_ != NCCL_COMM_NULL) {
      NCCL_CALL(ncclCommDeregister(comm_, handle_));
      comm_ = NCCL_COMM_NULL;
      handle_ = nullptr;
    }
  }

  ~reghandle() { destroy(); }

  reghandle(reghandle &&o) noexcept : comm_(o.comm_), handle_(o.handle_) {
    o.handle_ = nullptr;
    o.comm_ = NCCL_COMM_NULL;
  }

  reghandle &operator=(reghandle &&o) noexcept {
    if (this != &o) {
      if (handle_ && comm_ != NCCL_COMM_NULL) {
        NCCL_CALL(ncclCommDeregister(comm_, handle_));
      }
      comm_ = o.comm_;
      handle_ = o.handle_;
      o.handle_ = nullptr;
      o.comm_ = NCCL_COMM_NULL;
    }
    return *this;
  }

  reghandle(const reghandle &) = delete;
  reghandle &operator=(const reghandle &) = delete;

  void *get() const { return handle_; }
  explicit operator bool() const { return handle_ != nullptr; }
};


class ncclcommview {
 protected:
  ncclComm_t nccl_comm_{NCCL_COMM_NULL};
  ncclcommview() = default;
  ncclcommview(ncclComm_t comm) : nccl_comm_(comm) {};

 public:
  ncclComm_t get() { return nccl_comm_; }
  void send(const void *sendbuff, size_t count, ncclDataType_t datatype,
            int peer, gcxx::StreamView stream) {
    NCCL_CALL(ncclSend(sendbuff, count, datatype, peer, nccl_comm_, stream));
  }
  void recv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer,
            gcxx::StreamView stream) {
    NCCL_CALL(ncclRecv(recvbuff, count, datatype, peer, nccl_comm_, stream));
  }
  void sendrecv(const void *sendbuff, void *recvbuff, size_t count,
                ncclDataType_t datatype, int sendpeer, int recvpeer,
                gcxx::StreamView stream) {
    nccl::groupstart();
    send(sendbuff, count, datatype, sendpeer, stream);
    recv(recvbuff, count, datatype, recvpeer, stream);
    nccl::groupend();
  }

  template <typename VT>
  [[nodiscard]] reghandle register_buffer(VT *buff, size_t numelems) {
    return reghandle(nccl_comm_, buff, numelems * sizeof(VT));
  }
};

class ncclcomm : public ncclcommview {
 public:
  ncclcomm(int rank, int size, ncclUniqueId id) {
    NCCL_CALL(ncclCommInitRank(&nccl_comm_, size, id, rank));
  }

  ~ncclcomm() { NCCL_CALL(ncclCommDestroy(nccl_comm_)); }
};