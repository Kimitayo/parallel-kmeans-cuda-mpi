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

    // inicializa com seed aleatoria
    srand(time(NULL));

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

void liberarCentroides(Centroide *centroides, int k) {
    for (int i = 0; i < k; i++) {
        free(centroides[i].features);
    }

    free(centroides);
}