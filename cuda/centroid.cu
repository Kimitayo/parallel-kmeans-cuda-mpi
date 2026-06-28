#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>

#include "centroid.cuh"
#include "cuda_utils.cuh"

void alocarCentroides(double **centroides, double **centroidesAntigos, double **somaCentroides, int **contagemClusters, int k, int numFeatures) {

    CUDA_CHECK(cudaMallocManaged(centroides, k * numFeatures * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(centroidesAntigos, k * numFeatures * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(somaCentroides, k * numFeatures * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(contagemClusters, k * sizeof(int)));
}

void inicializarCentroides(const double *features, double *centroides, int numSamples, int numFeatures, int k) {

    srand((unsigned int) time(NULL));

    for (int c = 0; c < k; c++) {

        int indice = rand() % numSamples;

        int inicioAmostra = indice * numFeatures;
        int inicioCentroide = c * numFeatures;

        for (int f = 0; f < numFeatures; f++) {
            centroides[inicioCentroide + f] = features[inicioAmostra + f];
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__
void kernelCopiarCentroides(const double *origem, double *destino, int totalValores) {

    int indice = indiceGlobal();

    if (indice >= totalValores)
        return;

    destino[indice] = origem[indice];
}

__global__
void kernelZerarSomaCentroides(double *somaCentroides, int totalValores) {

    int indice = indiceGlobal();

    if (indice >= totalValores)
        return;

    somaCentroides[indice] = 0.0;
}

__global__
void kernelZerarContagemClusters(int *contagemClusters, int k) {

    int indice = indiceGlobal();

    if (indice >= k)
        return;

    contagemClusters[indice] = 0;
}

__global__
void kernelSomarCentroides(const double *features, const int *clusters, double *somaCentroides, int *contagemClusters, int numSamples, int numFeatures) {

    int amostra = indiceGlobal();

    if (amostra >= numSamples)
        return;

    int cluster = clusters[amostra];

    atomicAdd(&contagemClusters[cluster], 1);

    int inicioAmostra = amostra * numFeatures;
    int inicioCluster = cluster * numFeatures;

    for (int f = 0; f < numFeatures; f++) {
        atomicAdd(&somaCentroides[inicioCluster + f], features[inicioAmostra + f]);
    }
}

__global__
void kernelAtualizarCentroides(double *centroides, double *somaCentroides, const int *contagemClusters, int k, int numFeatures) {

    int indice = indiceGlobal();

    int totalValores = k * numFeatures;

    if (indice >= totalValores)
        return;

    int cluster = indice / numFeatures;

    if (contagemClusters[cluster] == 0)
        return;

    centroides[indice] = somaCentroides[indice] / contagemClusters[cluster];
}

void copiarCentroides(const double *origem, double *destino, int k, int numFeatures) {

    int totalValores = k * numFeatures;

    int blocos = calcularQuantidadeBlocos(totalValores, THREADS_POR_BLOCO);

    kernelCopiarCentroides<<<blocos, THREADS_POR_BLOCO>>>(origem, destino, totalValores);

    CUDA_KERNEL_CHECK();
}

void atualizarCentroides(const double *features, const int *clusters, double *centroides, double *somaCentroides, int *contagemClusters, int numSamples, int numFeatures, int k) {

    int totalValores = k * numFeatures;

    int blocosCentroides = calcularQuantidadeBlocos(totalValores, THREADS_POR_BLOCO);
    int blocosClusters = calcularQuantidadeBlocos(k, THREADS_POR_BLOCO);
    int blocosAmostras = calcularQuantidadeBlocos(numSamples, THREADS_POR_BLOCO);

    kernelZerarSomaCentroides<<<blocosCentroides, THREADS_POR_BLOCO>>>(somaCentroides, totalValores);
    CUDA_KERNEL_CHECK();

    kernelZerarContagemClusters<<<blocosClusters, THREADS_POR_BLOCO>>>(contagemClusters, k);
    CUDA_KERNEL_CHECK();

    kernelSomarCentroides<<<blocosAmostras, THREADS_POR_BLOCO>>>(features, clusters, somaCentroides, contagemClusters, numSamples, numFeatures);
    CUDA_KERNEL_CHECK();

    kernelAtualizarCentroides<<<blocosCentroides, THREADS_POR_BLOCO>>>(centroides, somaCentroides, contagemClusters, k, numFeatures);
    CUDA_KERNEL_CHECK();
}

void liberarCentroides(double *centroides, double *centroidesAntigos, double *somaCentroides, int *contagemClusters) {

    CUDA_CHECK(cudaFree(centroides));
    CUDA_CHECK(cudaFree(centroidesAntigos));
    CUDA_CHECK(cudaFree(somaCentroides));
    CUDA_CHECK(cudaFree(contagemClusters));
}