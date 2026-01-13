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

inline int rows_in_rank(int rank, int size, int N)
{
  int basecount = N / size;
  int remainder = N % size;
  return basecount + ((remainder > rank) ? 1 : 0);
}

inline int rows_start_of_rank(int rank, int size, int N)
{
  int basecount = N / size;
  int remainder = N % size;
  return (basecount * rank) + std::min(rank, remainder);
}