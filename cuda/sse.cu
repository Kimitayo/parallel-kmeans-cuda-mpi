#include <cuda_runtime.h>

#include "sse.cuh"
#include "cuda_utils.cuh"
#include "distance.cuh"

__global__
void kernelCalcularSSE(const double *features, const double *centroides, const int *clusters, double *sse, int numSamples, int numFeatures) {

    int amostra = indiceGlobal();

    if (amostra >= numSamples)
        return;

    int cluster = clusters[amostra];

    const double *ponto = &features[amostra * numFeatures];
    const double *centroide = &centroides[cluster * numFeatures];

    double distancia = distanciaEuclidiana(ponto, centroide, numFeatures);

    atomicAdd(sse, distancia);
}

double calcularSSE(const double *features, const double *centroides, const int *clusters, double *bufferSSE, int numSamples, int numFeatures) {

    *bufferSSE = 0.0;

    int blocos = calcularQuantidadeBlocos(numSamples, THREADS_POR_BLOCO);

    kernelCalcularSSE<<<blocos, THREADS_POR_BLOCO>>>(features, centroides, clusters, bufferSSE, numSamples, numFeatures);

    CUDA_KERNEL_CHECK();

    return *bufferSSE;
}