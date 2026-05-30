#include <stdio.h>
#include "normalize.h"

void normalizarDataset(Dataset *dataset) {

    int linhas = dataset->linhas;
    int colunas = dataset->colunas;

    // os vetores pra guardar os min e max
    double min[colunas];
    double max[colunas];

    // inicializacao dos min e max
    for (int j = 0; j < colunas; j++) {

        min[j] = dataset->dados[0].features[j];
        max[j] = dataset->dados[0].features[j];
    }

    // pra encontrar os min e max
    for (int i = 0; i < linhas; i++) {

        for (int j = 0; j < colunas; j++) {

            double valor =
                dataset->dados[i].features[j];

            if (valor < min[j]) {
                min[j] = valor;
            }

            if (valor > max[j]) {
                max[j] = valor;
            }
        }
    }

    // aplicar a nomralizacao
    for (int i = 0; i < linhas; i++) {

        for (int j = 0; j < colunas; j++) {

            double valor =
                dataset->dados[i].features[j];

            // Evita divisão por zero
            if (max[j] - min[j] != 0) {

                dataset->dados[i].features[j] =
                    (valor - min[j]) /
                    (max[j] - min[j]);
            }
        }
    }

    printf("Dataset normalizado com sucesso.\n");
}