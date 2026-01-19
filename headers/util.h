#pragma once

#include <fmt/format.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string_view>


#ifdef __NVCC__
#define FCT_DECORATOR __host__ __device__
#else
#define FCT_DECORATOR
#endif


template <typename tpe>
void printStats(const std::chrono::duration<double> elapsedSeconds, size_t nIt,
                size_t nCells, std::string_view tpeName, size_t numBytesPerCell,
                size_t numFlopsPerCell) {
  fmt::print("  #cells / #it:  {} / {}\n", nCells, nIt);
  fmt::print("  type:          {}\n", tpeName);
  fmt::print("  elapsed time:  {:.3f} ms\n", 1e3 * elapsedSeconds.count());
  fmt::print("  per iteration: {:.3f} ms\n",
             1e3 * elapsedSeconds.count() / nIt);
  fmt::print("  MLUP/s:        {:.3f}\n",
             1e-6 * nCells * nIt / elapsedSeconds.count());
  fmt::print("  bandwidth:     {:.3f} GB/s\n",
             1e-9 * numBytesPerCell * nCells * nIt / elapsedSeconds.count());
  fmt::print("  compute:       {:.3f} GFLOP/s\n",
             1e-9 * numFlopsPerCell * nCells * nIt / elapsedSeconds.count());
}

FCT_DECORATOR size_t ceilingDivide(size_t a, size_t b) {
  return (a + b - 1) / b;
}

FCT_DECORATOR size_t ceilToMultipleOf(size_t a, size_t b) {
  return ceilingDivide(a, b) * b;
}

inline int rows_in_rank(int rank, int size, int N) {
  int basecount = N / size;
  int remainder = N % size;
  return basecount + ((remainder > rank) ? 1 : 0);
}

inline int get_rank_below(const int rank, const int size) {
  return rank - 1;
}

inline int get_rank_above(const int rank, const int size) {
  return rank + 1;
}

inline bool is_top_check(const int rank, const int size) {
  return (rank == (size - 1));
}


inline bool is_bottom_check(const int rank, const int size) {
  return (rank == 0);
}

inline bool is_central_check(const int rank, const int size) {
  return !(is_top_check(rank, size) || is_bottom_check(rank, size));
}

inline int chunk_rows_of_rank(int rank, int size, int N) {
  if (size == 1)
    return N;
  const bool is_top = is_top_check(rank, size);
  const bool is_bottom = is_bottom_check(rank, size);
  const bool is_central = is_central_check(rank, size);
  return rows_in_rank(rank, size, N) + is_top + is_bottom + 2 * is_central;
}

inline int rows_start_of_rank(int rank, int size, int N) {
  int basecount = N / size;
  int remainder = N % size;
  return (basecount * rank) + std::min(rank, remainder);
}