#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <float.h>

#include "tipos.h"
#include "ingest.h"
#include "normalize.h"
#include "centroid.h"
#include "cluster.h"
#include "convergence.h"
#include "sse.h"

int K_CLUSTERS = 3;
int MAX_ITER_ACOES = 100;
int NUM_FEATURES = 11;

static void copiarDatasetParaBuffers(Dataset *dataset, Centroide *centroides, double *features, int *clusters, double *centroidesFlat) {
    int numVinhos = dataset->linhas;
    int numFeatures = dataset->colunas;

    for (int i = 0; i < numVinhos; i++) {
        clusters[i] = dataset->dados[i].cluster;

        for (int f = 0; f < numFeatures; f++) {
            features[i * numFeatures + f] = dataset->dados[i].features[f];
        }
    }

    for (int c = 0; c < K_CLUSTERS; c++) {
        for (int f = 0; f < numFeatures; f++) {
            centroidesFlat[c * numFeatures + f] = centroides[c].features[f];
        }
    }
}

static void copiarBuffersParaEstruturas(Dataset *dataset, Centroide *centroides, const int *clusters, const double *centroidesFlat) {
    int numVinhos = dataset->linhas;
    int numFeatures = dataset->colunas;

    for (int i = 0; i < numVinhos; i++) {
        dataset->dados[i].cluster = clusters[i];
    }

    for (int c = 0; c < K_CLUSTERS; c++) {
        for (int f = 0; f < numFeatures; f++) {
            centroides[c].features[f] = centroidesFlat[c * numFeatures + f];
        }
    }
}

void liberarDataset(Dataset *dataset) {
    if (dataset) {
        if (dataset->dados) {
            for (int i = 0; i < dataset->linhas; i++) {
                free(dataset->dados[i].features);
            }

            free(dataset->dados);
        }

        free(dataset);
    }
}

void executarKmeans(Dataset *dataset, Centroide *centroides) {
    int num_vinhos = dataset->linhas;
    int k = K_CLUSTERS;
    int features = NUM_FEATURES;
    int totalFeatures = num_vinhos * features;
    int totalCentroides = k * features;

    double *featuresFlat = (double*) malloc(totalFeatures * sizeof(double));
    int *clusters = (int*) malloc(num_vinhos * sizeof(int));
    double *centroidesFlat = (double*) malloc(totalCentroides * sizeof(double));
    double *centroidesAntigos = (double*) malloc(totalCentroides * sizeof(double));
    double *somaCentroides = (double*) malloc(totalCentroides * sizeof(double));
    int *contagemClusters = (int*) malloc(k * sizeof(int));

    if (!featuresFlat || !clusters || !centroidesFlat || !centroidesAntigos || !somaCentroides || !contagemClusters) {
        printf("Erro ao alocar buffers contiguos para GPU.\n");
        free(featuresFlat);
        free(clusters);
        free(centroidesFlat);
        free(centroidesAntigos);
        free(somaCentroides);
        free(contagemClusters);
        exit(1);
    }

    copiarDatasetParaBuffers(dataset, centroides, featuresFlat, clusters, centroidesFlat);

    #pragma omp target data map(to: featuresFlat[0:totalFeatures]) \
                            map(tofrom: clusters[0:num_vinhos], centroidesFlat[0:totalCentroides]) \
                            map(alloc: centroidesAntigos[0:totalCentroides], somaCentroides[0:totalCentroides], contagemClusters[0:k])
    {
        for (int iter = 0; iter < MAX_ITER_ACOES; iter++) {
            int mudouCluster = 0;
            double diferencaTotal = 0.0;

            #pragma omp target teams distribute parallel for
            for (int i = 0; i < totalCentroides; i++) {
                centroidesAntigos[i] = centroidesFlat[i];
                somaCentroides[i] = 0.0;
            }

            #pragma omp target teams distribute parallel for
            for (int c = 0; c < k; c++) {
                contagemClusters[c] = 0;
            }

            #pragma omp target teams distribute parallel for reduction(|:mudouCluster)
            for (int i = 0; i < num_vinhos; i++) {
                double menorDistancia = DBL_MAX;
                int melhorCluster = 0;
                int inicioAmostra = i * features;

                for (int c = 0; c < k; c++) {
                    double distancia = 0.0;
                    int inicioCentroide = c * features;

                    for (int f = 0; f < features; f++) {
                        double diferenca = featuresFlat[inicioAmostra + f] - centroidesFlat[inicioCentroide + f];
                        distancia += diferenca * diferenca;
                    }

                    if (distancia < menorDistancia) {
                        menorDistancia = distancia;
                        melhorCluster = c;
                    }
                }

                if (clusters[i] != melhorCluster) {
                    mudouCluster = 1;
                }

                clusters[i] = melhorCluster;

                #pragma omp atomic update
                contagemClusters[melhorCluster]++;

                for (int f = 0; f < features; f++) {
                    #pragma omp atomic update
                    somaCentroides[melhorCluster * features + f] += featuresFlat[inicioAmostra + f];
                }
            }

            #pragma omp target teams distribute parallel for
            for (int i = 0; i < totalCentroides; i++) {
                int cluster = i / features;

                if (contagemClusters[cluster] > 0) {
                    centroidesFlat[i] = somaCentroides[i] / contagemClusters[cluster];
                }
            }

            #pragma omp target teams distribute parallel for reduction(+:diferencaTotal)
            for (int i = 0; i < totalCentroides; i++) {
                double diferenca = centroidesAntigos[i] - centroidesFlat[i];
                diferencaTotal += diferenca < 0.0 ? -diferenca : diferenca;
            }

            if (!mudouCluster || diferencaTotal < 0.0001) {
                printf("\nConvergiu na iteracao %d\n", iter + 1);
                break;
            }
        }
    }

    copiarBuffersParaEstruturas(dataset, centroides, clusters, centroidesFlat);

    free(featuresFlat);
    free(clusters);
    free(centroidesFlat);
    free(centroidesAntigos);
    free(somaCentroides);
    free(contagemClusters);
}

int main(int argc, char *argv[]) {
    const char *nomeArquivo = "../vinhos.csv";

    if (argc == 2) {
        K_CLUSTERS = atoi(argv[1]);
    } else if (argc == 3) {
        K_CLUSTERS = atoi(argv[1]);
        MAX_ITER_ACOES = atoi(argv[2]);
    }

    double inicio = omp_get_wtime();

    Dataset *dataset = carregarDados(nomeArquivo);
    NUM_FEATURES = dataset->colunas;

    Vinho *dados = dataset->dados;

    normalizarDataset(dataset);

    Centroide *centroides = inicializarCentroides(dataset, K_CLUSTERS);

    executarKmeans(dataset, centroides);

    double fim = omp_get_wtime();
    printf("\nTempo de execucao: %.6f segundos\n", fim - inicio);

    printf("\nCentroides finais:\n\n");
    for (int i = 0; i < K_CLUSTERS; i++) {
        printf("Centroide %d:\n", i);

        for (int j = 0; j < NUM_FEATURES; j++) {
            printf("%.3f ", centroides[i].features[j]);
        }

        printf("\n\n");
    }

    printf("\nPrimeiros vinhos agrupados:\n\n");

    for (int i = 0; i < 10; i++) {
        printf("Vinho %d -> Cluster %d\n", i, dataset->dados[i].cluster);
    }

    int contagem[K_CLUSTERS];

    memset(contagem, 0, sizeof(contagem));

    for (int i = 0; i < dataset->linhas; i++) {
        contagem[dados[i].cluster]++;
    }

    printf("\nDistribuicao dos clusters:\n");

    for (int i = 0; i < K_CLUSTERS; i++) {
        printf("Cluster %d: %d vinhos\n", i, contagem[i]);
    }

    double sse = calcularSSE(dataset, centroides);
    printf("\nQualidade do agrupamento (SSE): %.6f\n", sse);
    double sseMedio = sse / dataset->linhas;
    printf("SSE medio: %.6f\n", sseMedio);

    liberarCentroides(centroides, K_CLUSTERS);
    liberarDataset(dataset);
    return 0;
}
