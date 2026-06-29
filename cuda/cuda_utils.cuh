#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

constexpr int THREADS_POR_BLOCO = 256;

// Retorna o índice global da thread.
__device__ __forceinline__
int indiceGlobal() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

int calcularQuantidadeBlocos(int elementos, int threadsPorBloco = 256);

void verificarCuda(cudaError_t erro, const char *arquivo, int linha);

void verificarKernel(const char *arquivo, int linha);

#define CUDA_CHECK(call) \
    verificarCuda((call), __FILE__, __LINE__)

#define CUDA_KERNEL_CHECK() \
    verificarKernel(__FILE__, __LINE__)

#endif