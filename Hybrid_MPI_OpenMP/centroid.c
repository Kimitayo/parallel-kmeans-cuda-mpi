#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "centroid.h"

Centroide* inicializarCentroides(Dataset *dataset, int k) {
    if (!dataset || !dataset->dados || dataset->linhas <= 0 || dataset->colunas <= 0 || k <= 0) {
        printf("Dataset ou quantidade de clusters invalida.\n");
        exit(1);
    }

    int numFeatures = dataset->colunas;

    Centroide *centroides = (Centroide*) calloc(k, sizeof(Centroide));

    if (!centroides) {
        printf("Erro ao alocar centroides\n");
        exit(1);
    }

    // inicializa com seed aleatoria
    srand(time(NULL));

    int *indicesAleatorios = malloc(k * sizeof(int));
    if (!indicesAleatorios) {
        printf("Erro ao alocar indices aleatorios dos centroides.\n");
        free(centroides);
        exit(1);
    }

    for (int i = 0; i < k; i++) {
        indicesAleatorios[i] = rand() % dataset->linhas;
    }

    int erroAlocacao = 0;

    // Para cada centroide
    #pragma omp parallel for reduction(|:erroAlocacao) schedule(static)
    for (int i = 0; i < k; i++) {
        centroides[i].features = (double*) malloc(numFeatures * sizeof(double));

        if (!centroides[i].features) {
            erroAlocacao = 1;
            continue;
        }

        // escolhe um vinho aleatorio
        int indiceAleatorio = indicesAleatorios[i];

        // Copia os atributos do vinho
        for (int j = 0; j < numFeatures; j++) {
            centroides[i].features[j] = dataset->dados[indiceAleatorio].features[j];
        }
    }

    free(indicesAleatorios);

    if (erroAlocacao) {
        printf("Erro ao alocar features do centroide.\n");
        for (int i = 0; i < k; i++) {
            free(centroides[i].features);
        }
        free(centroides);
        exit(1);
    }

    return centroides;
}

Centroide* copiarCentroides(Centroide *originais, int k, int numFeatures) {
    Centroide *copia = calloc(k, sizeof(Centroide));

    // verificar a alocacao
    if (!copia) {
        printf("Erro ao alocar copia dos centroides.\n");
        exit(1);
    }

    int erroAlocacao = 0;

    #pragma omp parallel for reduction(|:erroAlocacao) schedule(static)
    for (int i = 0; i < k; i++) {
        copia[i].features = malloc(numFeatures * sizeof(double));

        // verificar a alocacao das features
        if (!copia[i].features) {
            erroAlocacao = 1;
            continue;
        }

        for (int j = 0; j < numFeatures; j++) {
            copia[i].features[j] = originais[i].features[j];
        }
    }

    if (erroAlocacao) {
        printf("Erro ao alocar features da copia.\n");
        for (int i = 0; i < k; i++) {
            free(copia[i].features);
        }
        free(copia);
        exit(1);
    }

    return copia;
}

void atualizarCentroides(Dataset *dataset, Centroide *centroides, int k) {
    if (!dataset || !dataset->dados || !centroides || k <= 0 || dataset->linhas <= 0 || dataset->colunas <= 0) {
        return;
    }

    int numFeatures = dataset->colunas;
    size_t totalFeatures = (size_t) k * (size_t) numFeatures;
    int numThreads = omp_get_max_threads();

    // quantidade de vinhos por cluster
    int *contagem = (int*) calloc(k, sizeof(int));
    double *soma = (double*) calloc(totalFeatures, sizeof(double));
    int *contagemThreads = (int*) calloc((size_t) numThreads * (size_t) k, sizeof(int));
    double *somaThreads = (double*) calloc((size_t) numThreads * totalFeatures, sizeof(double));

    if (!contagem || !soma || !contagemThreads || !somaThreads) {
        printf("Erro ao alocar acumuladores dos centroides.\n");
        free(contagem);
        free(soma);
        free(contagemThreads);
        free(somaThreads);
        exit(1);
    }

    // soma features dos vinhos
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int *contagemLocal = contagemThreads + ((size_t) tid * (size_t) k);
        double *somaLocal = somaThreads + ((size_t) tid * totalFeatures);

        #pragma omp for schedule(static)
        for (int i = 0; i < dataset->linhas; i++) {
            int cluster = dataset->dados[i].cluster;

            if (cluster < 0 || cluster >= k) {
                continue; // ignora vinhos sem cluster valido
            }

            contagemLocal[cluster]++;

            for (int f = 0; f < numFeatures; f++) {
                somaLocal[cluster * numFeatures + f] += dataset->dados[i].features[f];
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < k; c++) {
        int totalContagem = 0;

        for (int t = 0; t < numThreads; t++) {
            totalContagem += contagemThreads[(size_t) t * (size_t) k + c];
        }

        contagem[c] = totalContagem;
    }

    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < totalFeatures; idx++) {
        double totalSoma = 0.0;

        for (int t = 0; t < numThreads; t++) {
            totalSoma += somaThreads[(size_t) t * totalFeatures + idx];
        }

        soma[idx] = totalSoma;
    }

    // divide pela quantidade; clusters vazios preservam o centroide anterior
    #pragma omp parallel for collapse(2) schedule(static)
    for (int c = 0; c < k; c++) {
        for (int f = 0; f < numFeatures; f++) {
            if (contagem[c] != 0) {
                centroides[c].features[f] = soma[c * numFeatures + f] / contagem[c];
            }
        }
    }

    free(contagem);
    free(soma);
    free(contagemThreads);
    free(somaThreads);
}

void liberarCentroides(Centroide *centroides, int k) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < k; i++) {
        free(centroides[i].features);
    }

    free(centroides);
}
