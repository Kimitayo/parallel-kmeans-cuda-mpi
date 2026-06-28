#include <stdio.h>
#include <cuda_runtime.h>

#include "flatten.cuh"

void flattenDataset(Dataset *dataset, double **features, int **clusters) {
    int numSamples = dataset->linhas;
    int numFeatures = dataset->colunas;

    cudaMallocManaged(features, numSamples * numFeatures * sizeof(double));

    cudaMallocManaged(clusters, numSamples * sizeof(int));

    for (int i = 0; i < numSamples; i++) {

        (*clusters)[i] = -1;

        for (int j = 0; j < numFeatures; j++) {
            (*features)[i * numFeatures + j] = dataset->dados[i].features[j];
        }
    }

    cudaDeviceSynchronize();
}

void liberarFlatten(double *features, int *clusters) {
    cudaFree(features);
    cudaFree(clusters);
}