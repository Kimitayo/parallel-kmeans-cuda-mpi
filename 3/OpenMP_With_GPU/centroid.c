#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "centroid.h"

Centroide* inicializarCentroides(Dataset *dataset, int k) {
    int numFeatures = dataset->colunas;

    Centroide *centroides = (Centroide*) malloc(k * sizeof(Centroide));

    if (!centroides) {
        printf("Erro ao alocar centroides\n");
        exit(1);
    }

    // semente fixa
    srand(42);

    // Para cada centroide
    for (int i = 0; i < k; i++) {
        centroides[i].features = (double*) malloc(numFeatures * sizeof(double));

        if (!centroides[i].features) {
            printf("Erro ao alocar features do centroide.\n");
            exit(1);
        }

        // escolhe um vinho aleatorio
        int indiceAleatorio = rand() % dataset->linhas;

        // Copia os atributos do vinho
        for (int j = 0; j < numFeatures; j++) {
            centroides[i].features[j] = dataset->dados[indiceAleatorio].features[j];
        }
    }
    return centroides;
}

Centroide* copiarCentroides(Centroide *originais, int k, int numFeatures) {
    Centroide *copia = malloc(k * sizeof(Centroide));

    // verificar a alocacao
    if (!copia) {
        printf("Erro ao alocar copia dos centroides.\n");
        exit(1);
    }

    for (int i = 0; i < k; i++) {
        copia[i].features = malloc(numFeatures * sizeof(double));

        // verificar a alocacao das features
        if (!copia[i].features) {
            printf("Erro ao alocar features da copia.\n");
            exit(1);
        }

        for (int j = 0; j < numFeatures; j++) {
            copia[i].features[j] = originais[i].features[j];
        }
    }
    return copia;
}

void atualizarCentroides(Dataset *dataset, Centroide *centroides, int k) {
    int numFeatures = dataset->colunas;
    int totalValores = k * numFeatures;

    // quantidade de vinhos por cluster
    int *contagem = (int*) calloc(k, sizeof(int));

    double *soma = (double*) calloc(totalValores, sizeof(double));

    if (!contagem || !soma) {
        printf("Erro ao alocar contagem/soma.\n");
        exit(1);
    }

    // Mapeamento real dos dados (nao só dos ponteiros) para a GPU
    #pragma omp target enter data map(to: contagem[0:k], soma[0:totalValores])

    // soma features dos vinhos
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < dataset->linhas; i++) {
        int cluster = dataset->dados[i].cluster;

        #pragma omp atomic update
        contagem[cluster]++;

        for (int f = 0; f < numFeatures; f++) {
            #pragma omp atomic update
            soma[cluster * numFeatures + f] += dataset->dados[i].features[f];
        }
    }

    // Traz soma/contagem de volta pra CPU antes de atualizar os centroides
    // (centroides[c].features é memoria normal da CPU, fora da regiao
    // target --> por isso essa parte roda em CPU, nao em GPU)
    #pragma omp target update from(contagem[0:k], soma[0:totalValores])

    // divide pela quantidade, clusters vazios preservam o centroide
    // anterior (sem zerar antes --> um centroide zerado, com dados
    // normalizados entre 0 e 1, pode atrair pontos artificialmente pra um
    // canto do espaco de features que nao tem nada a ver com os dados reais)
    for (int c = 0; c < k; c++) {
        if (contagem[c] == 0)
            continue;

        for (int f = 0; f < numFeatures; f++) {
            centroides[c].features[f] = soma[c * numFeatures + f] / contagem[c];
        }
    }

    #pragma omp target exit data map(delete: contagem[0:k], soma[0:totalValores])

    free(contagem);
    free(soma);
}

void liberarCentroides(Centroide *centroides, int k) {
    for (int i = 0; i < k; i++) {
        free(centroides[i].features);
    }

    free(centroides);
}
