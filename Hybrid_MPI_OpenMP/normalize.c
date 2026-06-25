#include <stdio.h>
#include<omp.h>
#include "normalize.h"

void normalizarDataset(Dataset *dataset) {

    int linhas = dataset->linhas;
    int colunas = dataset->colunas;

    // os vetores pra guardar os min e max
    double min[colunas];
    double max[colunas];

    // Encontra min e max por coluna sem escrita concorrente.
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < colunas; j++) {
        double minLocal = dataset->dados[0].features[j];
        double maxLocal = dataset->dados[0].features[j];

        for (int i = 1; i < linhas; i++) {
            double valor = dataset->dados[i].features[j];

            if (valor < minLocal) minLocal = valor;
            if (valor > maxLocal) maxLocal = valor;
        }

        min[j] = minLocal;
        max[j] = maxLocal;
    }

    // Aplicando a normalizacao de forma paralela
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < linhas; i++) {
        for (int j = 0; j < colunas; j++) {

            double valor = dataset->dados[i].features[j];
            double amplitude = max[j] - min[j];
            // Evita divisão por zero
            if (amplitude > 0) {

                dataset->dados[i].features[j] = (valor - min[j]) / (amplitude);
            }else{
                dataset->dados[i].features[j] = 0.0;
            }
        }
    }

    printf("Dataset normalizado com sucesso.\n");
}
