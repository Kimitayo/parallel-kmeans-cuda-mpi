#include <stdio.h>
#include <cuda_runtime.h>

#include "normalize.cuh"

__global__
void kernelEncontrarMinMax(const double *features, double *minimos, double *maximos, int numSamples, int numFeatures) {

    int feature = blockIdx.x * blockDim.x + threadIdx.x;

    if (feature >= numFeatures)
        return;

    double minimo = features[feature];
    double maximo = features[feature];

    for (int i = 1; i < numSamples; i++) {

        double valor = features[i * numFeatures + feature];

        if (valor < minimo)
            minimo = valor;

        if (valor > maximo)
            maximo = valor;
    }

    minimos[feature] = minimo;
    maximos[feature] = maximo;
}

__global__
void kernelNormalizar(double *features, const double *minimos, const double *maximos, int numSamples, int numFeatures) {

    int indice = blockIdx.x * blockDim.x + threadIdx.x;

    if (indice >= numSamples)
        return;

    int inicio = indice * numFeatures;

    for (int f = 0; f < numFeatures; f++) {

        double minimo = minimos[f];
        double maximo = maximos[f];

        if (maximo - minimo > 0.0) {
            features[inicio + f] = (features[inicio + f] - minimo) / (maximo - minimo);

        } else {
            features[inicio + f] = 0.0;
        }
    }
}

void normalizarDataset(double *features, int numSamples, int numFeatures) {

    double *minimos;
    double *maximos;

    cudaMallocManaged(&minimos, numFeatures * sizeof(double));
    cudaMallocManaged(&maximos, numFeatures * sizeof(double));

    int threadsMinMax = 256;
    int blocosMinMax = (numFeatures + threadsMinMax - 1) / threadsMinMax;

    kernelEncontrarMinMax<<<blocosMinMax,threadsMinMax>>>(features, minimos, maximos, numSamples, numFeatures);

    cudaDeviceSynchronize();

    int threads = 256;

    int blocos = (numSamples + threads - 1) / threads;

    kernelNormalizar<<<blocos,threads>>>(features, minimos, maximos, numSamples, numFeatures);

    cudaDeviceSynchronize();

    cudaFree(minimos);
    cudaFree(maximos);

    printf("Dataset normalizado com sucesso.\n");
}