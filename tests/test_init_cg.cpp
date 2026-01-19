#include <iostream>
#include <cmath>
#include <vector>
#include "mpi_comm.hpp"
#include "util.h"
#include "cg-util.h"


// Test function
template <typename tpe>
bool compareArrays(const tpe* a, const tpe* b, size_t n, tpe tolerance = 1e-12) {
  for (size_t i = 0; i < n; ++i) {
    if (std::abs(a[i] - b[i]) > tolerance) {
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  mpienv env(argc, argv);
  
  mpicommview comm(MPI_COMM_WORLD);
  int rank = comm.rank();
  int size = comm.size();
  
  const size_t nx = 100;
  const size_t ny = 100;
  
  // Test both app modes
  for (int test_app = 0; test_app <= 1; ++test_app) {
    // app = test_app;
    
    // Serial version
    std::vector<double> u_serial(nx * ny);
    std::vector<double> rhs_serial(nx * ny);
    
    if (rank == 0) {
      initConjugateGradient(u_serial.data(), rhs_serial.data(), nx, ny);
    }
    
    // Broadcast serial results to all ranks for comparison
    MPI_Bcast(u_serial.data(), nx * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(rhs_serial.data(), nx * ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Distributed version
    auto local_ny = rows_in_rank(rank, size, ny);
    size_t local_size = (local_ny + 2) * nx; // +2 for halo rows
    std::vector<double> u_dist(local_size);
    std::vector<double> rhs_dist(local_size);
    
    initConjugateGradientDistributed(u_dist.data(), rhs_dist.data(), 
                                     nx, ny, comm);
    
    // Gather distributed results to rank 0
    std::vector<double> u_gathered;
    std::vector<double> rhs_gathered;
    
    if (rank == 0) {
      u_gathered.resize(nx * ny);
      rhs_gathered.resize(nx * ny);
    }
    
    // Calculate offsets for gathering
    auto y_start = rows_start_of_rank(rank, size, ny);
    bool is_bottom = is_bottom_check(rank, size);
    
    double* u_send = is_bottom ? u_dist.data() : u_dist.data() + nx;
    double* rhs_send = is_bottom ? rhs_dist.data() : rhs_dist.data() + nx;
    
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    for (int r = 0; r < size; ++r) {
      recvcounts[r] = rows_in_rank(r, size, ny) * nx;
      displs[r] = rows_start_of_rank(r, size, ny) * nx;
    }
    
    MPI_Gatherv(u_send, local_ny * nx, MPI_DOUBLE,
                u_gathered.data(), recvcounts.data(), displs.data(), 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Gatherv(rhs_send, local_ny * nx, MPI_DOUBLE,
                rhs_gathered.data(), recvcounts.data(), displs.data(), 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Compare results on rank 0
    if (rank == 0) {
      bool u_match = compareArrays(u_serial.data(), u_gathered.data(), nx * ny);
      bool rhs_match = compareArrays(rhs_serial.data(), rhs_gathered.data(), nx * ny);
      
      std::cout << "Test app=" << test_app << std::endl;
      std::cout << "  u arrays match: " << (u_match ? "YES" : "NO") << std::endl;
      std::cout << "  rhs arrays match: " << (rhs_match ? "YES" : "NO") << std::endl;
      
      if (!u_match || !rhs_match) {
        std::cout << "  First few differences:" << std::endl;
        int count = 0;
        for (size_t i = 0; i < nx * ny && count < 10; ++i) {
          if (std::abs(u_serial[i] - u_gathered[i]) > 1e-12 ||
              std::abs(rhs_serial[i] - rhs_gathered[i]) > 1e-12) {
            std::cout << "    idx=" << i << " serial_u=" << u_serial[i] 
                     << " dist_u=" << u_gathered[i]
                     << " serial_rhs=" << rhs_serial[i]
                     << " dist_rhs=" << rhs_gathered[i] << std::endl;
            count++;
          }
        }
      }
      std::cout << std::endl;
    }
  }
  
  return 0;
}
