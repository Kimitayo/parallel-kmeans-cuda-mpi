#include <float.h>

#include <cuda_runtime.h>

#include "cluster.cuh"
#include "cuda_utils.cuh"
#include "distance.cuh"

__global__
void kernelAtribuirClusters(const double *features, const double *centroides, int *clusters, int numSamples, int numFeatures, int k) {

    int amostra = indiceGlobal();

    if (amostra >= numSamples)
        return;

    double menorDistancia = DBL_MAX;
    int melhorCluster = 0;

    const double *ponto = &features[amostra * numFeatures];

    for (int c = 0; c < k; c++) {

        const double *centroide = &centroides[c * numFeatures];

        double distancia = distanciaEuclidiana(ponto, centroide, numFeatures);

        if (distancia < menorDistancia) {
            menorDistancia = distancia;
            melhorCluster = c;
        }
    }

    clusters[amostra] = melhorCluster;
}

void atribuirClusters(const double *features, const double *centroides, int *clusters, int numSamples, int numFeatures, int k) {

    int blocos = calcularQuantidadeBlocos(numSamples, THREADS_POR_BLOCO);

    kernelAtribuirClusters<<<blocos, THREADS_POR_BLOCO>>>(features, centroides, clusters, numSamples, numFeatures, k);

    CUDA_KERNEL_CHECK();
}