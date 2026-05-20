# Parallel K-Means Clustering: CPU & GPU Performance Optimization

A high-performance computing (HPC) project focused on accelerating the **K-Means clustering algorithm** across multiple parallel programming paradigms. The implementations are rigorously benchmarked and analyzed using the **Wine Dataset** to evaluate scalability, speedup, and hardware efficiency.

---

## 🚀 Project Overview

Clustering datasets efficiently requires leveraging modern multi-core CPUs and massively parallel GPUs. This project transitions a sequential K-Means baseline into highly optimized parallel versions, comparing distributed memory, shared memory, and native GPU execution.

### 📊 The Workload: Wine Dataset
The implementations utilize the **Wine Dataset**, which consists of chemical profiles of wines. Clustering this dataset serves as a realistic workload to benchmark algorithmic efficiency, memory access patterns, and data distribution across parallel nodes.

---

## 🛠️ Parallel Architectures Developed

The project contains four distinct versions developed in **C**:

1. **Sequential Baseline:** Single-threaded execution used as the performance anchor for all speedup calculations.
2. **Hybrid MPI + OpenMP:** Distributed and shared memory parallelization targeting CPU clusters. Focuses on domain decomposition, process synchronization, and minimizing communication overhead.
3. **OpenMP GPU Offloading:** Accelerated execution utilizing compiler directives to offload intensive compute workloads to the GPU.
4. **Native CUDA:** Low-level, fine-grained GPU optimization maximizing memory throughput (shared memory utilization, coalesced reads) and thread block configuration.

---

## 📉 Performance Evaluation & Engineering Metrics

Rather than just writing parallel code, this project focuses heavily on **quantitative performance analysis**. The implementations are evaluated under the following engineering criteria:

* **Execution Time & Speedup:** Measuring absolute time reduction relative to the sequential baseline.
* **Hardware Efficiency:** Evaluating how effectively the computing resources are utilized as core/thread counts scale.
* **Scalability Profiles:**
  * **Strong Scalability:** Keeping the workload fixed while varying the number of processes and threads.
  * **Weak Scalability:** Scaling the workload size proportionally to the processing power.
* **Architectural Bottlenecks:** Deep-dive analysis into MPI communication overhead, load balancing issues, and CPU-GPU data transfer latencies over the PCIe bus.

---

## 📊 Benchmarks & Results

> 💡 **Performance Note:** This section will be populated with scalability curves, speedup graphs, and comparative charts mapping CPU vs. GPU performance profiles once experimental runs on the cluster are completed.

| Architecture | Execution Time (ms) | Speedup | Efficiency |
| :--- | :--- | :--- | :--- |
| **Sequential Baseline** | *TBD* | 1.0x | 100% |
| **Hybrid MPI + OpenMP** | *TBD* | *TBD* | *TBD* |
| **OpenMP GPU Offloading** | *TBD* | *TBD* | *TBD* |
| **Native CUDA** | *TBD* | *TBD* | *TBD* |

---

## 📦 Requirements & Compilation

To compile and run these implementations, the following environment is required:
* A modern C/C++ compiler (GCC/Clang)
* An MPI implementation (e.g., OpenMPI, MPICH)
* OpenMP 4.5+ capable compiler for GPU offloading
* NVIDIA CUDA Toolkit
