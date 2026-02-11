# Conjugate Gradient (CG) Solver – CUDA Graph Optimized

This repository contains an optimized implementation of the Conjugate Gradient (CG) solver using CUDA Graphs and additional performance optimization techniques.

## Branches

There are two main development branches with updated implementations:

### 1. `multi_gpu_fullgraph`

- Multi-GPU implementation of the CG solver.
- Uses CUDA Graphs to capture and replay the full computation graph.
- Incorporates several optimization techniques for improved scalability and performance.
- Designed for distributed and multi-device environments.

### 2. `single_gpu_fullgraph`

- Single-GPU implementation of the CG solver.
- Uses CUDA Graphs for end-to-end execution.
- Includes kernel fusion, reduced launch overhead, and memory access optimizations.
- Implements residual convergence checking entirely on the GPU using CUDA Graph conditional nodes.
- Eliminates frequent host-device synchronization for improved performance.

## Key Features

- CUDA Graph-based execution
- Reduced kernel launch overhead
- Improved memory access patterns
- Optimized parallel redu
