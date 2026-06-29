#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "tipos.h"
#include "ingest.cuh"
#include "flatten.cuh"
#include "normalize.cuh"
#include "centroid.cuh"
#include "cluster.cuh"
#include "convergence.cuh"
#include "sse.cuh"
#include "cuda_utils.cuh"

int K_CLUSTERS = 3;
int MAX_ITER_ACOES = 100;
int NUM_FEATURES = 11;

void liberarDataset(Dataset *dataset) {

    if (dataset == NULL)
        return;

    if (dataset->dados != NULL) {

        for (int i = 0; i < dataset->linhas; i++) {
            free(dataset->dados[i].features);
        }

        free(dataset->dados);
    }

    free(dataset);
}

int main(int argc, char *argv[]) {

    const char *nomeArquivo = "../vinhos.csv";

    if (argc == 2) {
        K_CLUSTERS = atoi(argv[1]);
    }
    else if (argc == 3) {
        K_CLUSTERS = atoi(argv[1]);
        MAX_ITER_ACOES = atoi(argv[2]);
    }

    cudaEvent_t inicio;
    cudaEvent_t fim;

    CUDA_CHECK(cudaEventCreate(&inicio));
    CUDA_CHECK(cudaEventCreate(&fim));

    CUDA_CHECK(cudaEventRecord(inicio));

    Dataset *dataset = carregarDados(nomeArquivo);

    NUM_FEATURES = dataset->colunas;

    double *features;
    int *clusters;

    flattenDataset(dataset, &features, &clusters);

    int numSamples = dataset->linhas;

    liberarDataset(dataset);

    normalizarDataset(features, numSamples, NUM_FEATURES);

    double *centroides;
    double *centroidesAntigos;
    double *somaCentroides;
    int *contagemClusters;

    alocarCentroides(&centroides, &centroidesAntigos, &somaCentroides, &contagemClusters, K_CLUSTERS, NUM_FEATURES);

    int *resultadoConvergencia;
    double *sse;

    CUDA_CHECK(cudaMallocManaged(&resultadoConvergencia, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&sse, sizeof(double)));

    inicializarCentroides(features, centroides, numSamples, NUM_FEATURES, K_CLUSTERS);

    for (int iter = 0; iter < MAX_ITER_ACOES; iter++) {

        copiarCentroides(centroides, centroidesAntigos, K_CLUSTERS, NUM_FEATURES);

        atribuirClusters(features, centroides, clusters, numSamples, NUM_FEATURES, K_CLUSTERS);

        atualizarCentroides(features, clusters, centroides, somaCentroides, contagemClusters, numSamples, NUM_FEATURES, K_CLUSTERS);

        if (convergiu(centroides, centroidesAntigos, resultadoConvergencia, K_CLUSTERS, NUM_FEATURES, 0.0001)) {

            printf("\nConvergiu na iteracao %d\n", iter + 1);
            break;
        }
    }

    CUDA_CHECK(cudaEventRecord(fim));
    CUDA_CHECK(cudaEventSynchronize(fim));

    float tempo;

    CUDA_CHECK(cudaEventElapsedTime(&tempo, inicio, fim));

    printf("\nTempo de execucao: %.6f segundos\n", tempo / 1000.0);

    printf("\nCentroides finais:\n\n");

    for (int c = 0; c < K_CLUSTERS; c++) {

        printf("Centroide %d:\n", c);

        for (int f = 0; f < NUM_FEATURES; f++) {
            printf("%.3f ", centroides[c * NUM_FEATURES + f]);
        }

        printf("\n\n");
    }

    printf("\nPrimeiras amostras agrupadas:\n\n");

    int quantidade = numSamples < 10 ? numSamples : 10;

    for (int i = 0; i < quantidade; i++) {
        printf("Amostra %d -> Cluster %d\n", i, clusters[i]);
    }

    int *contagem = (int *) calloc(K_CLUSTERS, sizeof(int));

    for (int i = 0; i < numSamples; i++) {
        contagem[clusters[i]]++;
    }

    printf("\nDistribuicao dos clusters:\n");

    for (int i = 0; i < K_CLUSTERS; i++) {
        printf("Cluster %d: %d amostras\n", i, contagem[i]);
    }

    free(contagem);

    double resultadoSSE = calcularSSE(features, centroides, clusters, sse, numSamples, NUM_FEATURES);

    printf("\nQualidade do agrupamento (SSE): %.6f\n", resultadoSSE);
    printf("SSE medio: %.6f\n", resultadoSSE / numSamples);

    CUDA_CHECK(cudaFree(resultadoConvergencia));
    CUDA_CHECK(cudaFree(sse));

    liberarCentroides(centroides, centroidesAntigos, somaCentroides, contagemClusters);

    liberarFlatten(features, clusters);

    CUDA_CHECK(cudaEventDestroy(inicio));
    CUDA_CHECK(cudaEventDestroy(fim));

    return 0;
}