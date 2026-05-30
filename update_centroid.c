#include <stdlib.h>
#include <stdio.h>

#include "update_centroid.h"

void atualizarCentroides(Dataset *dataset, Centroide *centroides, int k) {
    int numFeatures = dataset->colunas;

    // quantidade de vinhos por cluster
    int *contagem = (int*) calloc(k, sizeof(int));

    if (!contagem) {
        printf("Erro ao alocar contagem.\n");
        exit(1);
    }

    // zera centroides
    for (int c = 0; c < k; c++) {
        for (int f = 0; f < numFeatures; f++) {
            centroides[c].features[f] = 0.0;
        }
    }

    // soma features dos vinhos
    for (int i = 0; i < dataset->linhas; i++) {
        int cluster = dataset->dados[i].cluster;
        contagem[cluster]++;

        for (int f = 0; f < numFeatures; f++) {
            centroides[cluster].features[f] += dataset->dados[i].features[f];
        }
    }

    // divide pela quantidade
    for (int c = 0; c < k; c++) {
        if (contagem[c] == 0)
            continue;

        for (int f = 0; f < numFeatures; f++) {
            centroides[c].features[f] /= contagem[c];
        }
    }

    free(contagem);
}