#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <cuda_runtime.h>

#include "cluster.h"

// ===========================================================================
// cluster.cu
//
// Mesma funcao "atribuirClusters" do cluster.c sequencial, com o MESMO
// comportamento e a MESMA assinatura. A unica diferenca e que o calculo
// de distancia (a parte mais pesada: para cada vinho, calcular a distancia
// para os k centroides) roda na GPU em paralelo, uma thread por vinho.
//
// Passos desta funcao (sempre os mesmos, em toda chamada):
//   1. "Achatar" o dataset (Dataset->dados[i].features, que sao varios
//      ponteiros) num unico vetor continuo de doubles > a GPU nao consegue
//      seguir ponteiros da CPU, so enxerga memoria contigua que foi
//      copiada explicitamente para ela.
//   2. Copiar esse vetor + os centroides para a GPU (cudaMemcpy).
//   3. Rodar o kernel (equivalente ao "for" do sequencial, mas paralelo).
//   4. Copiar o resultado (cluster de cada vinho) de volta para a CPU.
//   5. Escrever o resultado de volta no Dataset, exatamente como o
//      sequencial faria.
//
// OBS: alocar/copiar a cada chamada tem um custo (overhead de transferencia
// CPU<->GPU). Fizemos essa escolha de proposito para manter o codigo simples
// e parecido com as outras versoes --> e esse overhead, alias, e um dos
// pontos que o enunciado pede pra discutir (impacto da transferencia de
// dados entre CPU e GPU).
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

// Kernel: cada thread cuida de UM vinho (mesmo papel da iteracao "i" do
// for sequencial). Calcula a distancia para os k centroides e guarda o
// indice do mais proximo.
__global__ void kernelAtribuirClusters(
    const double *features,
    const double *centroides,
    int *clusterAtual,
    int linhas,
    int colunas,
    int k,
    int *houveMudanca
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= linhas) return;

    double menorDistancia = DBL_MAX;
    int melhorCluster = -1;

    for (int c = 0; c < k; c++) {
        double soma = 0.0;

        for (int f = 0; f < colunas; f++) {
            double diff = features[(size_t) i * colunas + f] - centroides[c * colunas + f];
            soma += diff * diff;
        }

        if (soma < menorDistancia) {
            menorDistancia = soma;
            melhorCluster = c;
        }
    }

    if (clusterAtual[i] != melhorCluster) {
        atomicExch(houveMudanca, 1);
    }

    clusterAtual[i] = melhorCluster;
}

int atribuirClusters(Dataset *dataset, Centroide *centroides, int k) {
    int linhas = dataset->linhas;
    int colunas = dataset->colunas;

    // 1. achatar dataset e centroides em vetores continuos
    double *h_features = (double *) malloc((size_t) linhas * colunas * sizeof(double));
    double *h_centroides = (double *) malloc((size_t) k * colunas * sizeof(double));
    int *h_cluster = (int *) malloc((size_t) linhas * sizeof(int));

    for (int i = 0; i < linhas; i++) {
        memcpy(h_features + (size_t) i * colunas, dataset->dados[i].features, colunas * sizeof(double));
        h_cluster[i] = dataset->dados[i].cluster;
    }

    for (int c = 0; c < k; c++) {
        memcpy(h_centroides + (size_t) c * colunas, centroides[c].features, colunas * sizeof(double));
    }

    // 2. copiar para a GPU 
    double *d_features, *d_centroides;
    int *d_cluster, *d_houveMudanca;

    CUDA_CHECK(cudaMalloc(&d_features, (size_t) linhas * colunas * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_centroides, (size_t) k * colunas * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cluster, (size_t) linhas * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_houveMudanca, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_features, h_features, (size_t) linhas * colunas * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroides, h_centroides, (size_t) k * colunas * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cluster, h_cluster, (size_t) linhas * sizeof(int), cudaMemcpyHostToDevice));

    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_houveMudanca, &zero, sizeof(int), cudaMemcpyHostToDevice));

    // 3. rodar o kernel (equivalente ao "for" do sequencial) 
    int threadsPorBloco = 256;
    int blocos = (linhas + threadsPorBloco - 1) / threadsPorBloco;

    kernelAtribuirClusters<<<blocos, threadsPorBloco>>>(
        d_features, d_centroides, d_cluster, linhas, colunas, k, d_houveMudanca);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. trazer o resultado de volta
    int houveMudanca;
    CUDA_CHECK(cudaMemcpy(h_cluster, d_cluster, (size_t) linhas * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&houveMudanca, d_houveMudanca, sizeof(int), cudaMemcpyDeviceToHost));

    // 5. escrever de volta no Dataset (igual ao sequencial)
    for (int i = 0; i < linhas; i++) {
        dataset->dados[i].cluster = h_cluster[i];
    }

    free(h_features);
    free(h_centroides);
    free(h_cluster);
    cudaFree(d_features);
    cudaFree(d_centroides);
    cudaFree(d_cluster);
    cudaFree(d_houveMudanca);

    return houveMudanca;
}
