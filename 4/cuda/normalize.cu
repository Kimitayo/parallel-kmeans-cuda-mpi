#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <cuda_runtime.h>

#include "normalize.h"

// ===========================================================================
// normalize.cu
//
// Mesma funcao "normalizarDataset" do normalize.c sequencial, mesma
// assinatura e mesmo resultado (normalizacao min-max por coluna). A
// diferenca e que o calculo do minimo/maximo de cada coluna e a divisao
// de cada valor rodam na GPU.
// ===========================================================================

#define CUDA_CHECK(chamada)                                                          \
    do {                                                                             \
        cudaError_t erro = (chamada);                                                \
        if (erro != cudaSuccess) {                                                   \
            fprintf(stderr, "Erro CUDA em %s:%d -> %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(erro));                                       \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

// Kernel 1: cada thread cuida de UMA coluna/feature (sao poucas: 11 no
// dataset de vinhos) e percorre todas as linhas procurando o min/max.
__global__ void kernelMinMax(
    const double *features,
    int linhas,
    int colunas,
    double *minimos,
    double *maximos
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= colunas) return;

    double menor = DBL_MAX;
    double maior = -DBL_MAX;

    for (int i = 0; i < linhas; i++) {
        double valor = features[(size_t) i * colunas + j];
        if (valor < menor) menor = valor;
        if (valor > maior) maior = valor;
    }

    minimos[j] = menor;
    maximos[j] = maior;
}

// Kernel 2: cada thread cuida de UM elemento (linha*coluna) e aplica a
// formula min-max, igual ao "for" duplo do sequencial.
__global__ void kernelNormalizar(
    double *features,
    int linhas,
    int colunas,
    const double *minimos,
    const double *maximos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = linhas * colunas;
    if (idx >= total) return;

    int j = idx % colunas;
    double amplitude = maximos[j] - minimos[j];

    if (amplitude > 0) {
        features[idx] = (features[idx] - minimos[j]) / amplitude;
    } else {
        features[idx] = 0.0;
    }
}

void normalizarDataset(Dataset *dataset) {
    int linhas = dataset->linhas;
    int colunas = dataset->colunas;

    // achatar features 
    double *h_features = (double *) malloc((size_t) linhas * colunas * sizeof(double));

    for (int i = 0; i < linhas; i++) {
        memcpy(h_features + (size_t) i * colunas, dataset->dados[i].features, colunas * sizeof(double));
    }

    // copiar para a GPU 
    double *d_features, *d_minimos, *d_maximos;

    CUDA_CHECK(cudaMalloc(&d_features, (size_t) linhas * colunas * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_minimos, (size_t) colunas * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_maximos, (size_t) colunas * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_features, h_features, (size_t) linhas * colunas * sizeof(double), cudaMemcpyHostToDevice));

    // rodar os kernels 
    int threadsPorBloco = 256;

    int blocosColunas = (colunas + threadsPorBloco - 1) / threadsPorBloco;
    kernelMinMax<<<blocosColunas, threadsPorBloco>>>(d_features, linhas, colunas, d_minimos, d_maximos);
    CUDA_CHECK(cudaGetLastError());

    int totalElementos = linhas * colunas;
    int blocosElementos = (totalElementos + threadsPorBloco - 1) / threadsPorBloco;
    kernelNormalizar<<<blocosElementos, threadsPorBloco>>>(d_features, linhas, colunas, d_minimos, d_maximos);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // trazer resultado de volta e escrever no Dataset 
    CUDA_CHECK(cudaMemcpy(h_features, d_features, (size_t) linhas * colunas * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < linhas; i++) {
        memcpy(dataset->dados[i].features, h_features + (size_t) i * colunas, colunas * sizeof(double));
    }

    free(h_features);
    cudaFree(d_features);
    cudaFree(d_minimos);
    cudaFree(d_maximos);

    printf("Dataset normalizado com sucesso.\n");
}
