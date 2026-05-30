#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include "tipos.h"
#include "ingest.h"
#include "normalize.h"
#include "distance.h"

//gcc kmeans.c ingest.c -o kmeans -lm

#define K_CLUSTERS 3 // Quantidade de grupos que queremos encontrar
#define MAX_ITER_ACOES 100 // Critério de parada por repetições

int NUM_FEATURES = 0;


void alterarVariavelGlobal(int numColunas) {
    NUM_FEATURES = numColunas;
}

void liberarDataset(Dataset *dataset) {
    if (dataset) {
        if (dataset->dados) {
            // Libera o array de features de cada vinho individualmente
            for (int i = 0; i < dataset->linhas; i++) {
                free(dataset->dados[i].features);
            }
            // Libera o array de vinhos
            free(dataset->dados);
        }
        // Libera a estrutura de controle
        free(dataset);
    }
}

int main(int argc, char *argv[]) {
    const char *nomeArquivo = "vinhos.csv"; 

    Dataset *dataset = carregarDados(nomeArquivo);

    // aplicar normalizaçao
    normalizarDataset(dataset);
    Vinho *dados = dataset->dados;
    
    int tamanhoDataset = dataset->linhas;
    alterarVariavelGlobal(dataset->colunas); // atualizar a variável global com o número de colunas do dataset

    // aplicar distance.h 
    double distancia = distanciaEuclidiana(dados[0].features, dados[1].features, NUM_FEATURES);

    printf("\nDistancia entre vinho 0 e vinho 1: %f\n", distancia);


    
    printf("Dados carregados: %d linhas e %d colunas (features) %d %d.\n", tamanhoDataset, NUM_FEATURES, dados[0].qualidade, dados[0].id);

    liberarDataset(dataset);
    return 0;
}
