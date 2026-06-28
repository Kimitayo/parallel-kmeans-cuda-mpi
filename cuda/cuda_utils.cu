#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "cuda_utils.cuh"

// Calcula a quantidade de blocos necessária para
// processar "elementos" utilizando "threadsPorBloco".
int calcularQuantidadeBlocos(int elementos, int threadsPorBloco) {
    return (elementos + threadsPorBloco - 1) / threadsPorBloco;
}

// Verifica erros de chamadas CUDA.
// Deve ser utilizada apenas em funções executadas na CPU.
void verificarCuda(cudaError_t erro, const char *arquivo, int linha) {

    if (erro != cudaSuccess) {

        fprintf(
            stderr,
            "Erro CUDA em %s:%d\n%s\n",
            arquivo,
            linha,
            cudaGetErrorString(erro)
        );

        exit(EXIT_FAILURE);
    }
}

// Macro para reduzir repetição.
// Exemplo:
// CUDA_CHECK(cudaMallocManaged(...));
#define CUDA_CHECK(call) \
    verificarCuda((call), __FILE__, __LINE__)

// Verifica erros ocorridos durante a execução
// do último kernel lançado.
void verificarKernel(const char *arquivo, int linha) {

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

#define CUDA_KERNEL_CHECK() \
    verificarKernel(__FILE__, __LINE__)
