#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tipos.h"
#include "ingest.h"
#include "normalize.h"
#include "distance.h"
#include "centroid.h"
#include "cluster.h"
#include "update_centroid.h"

//gcc kmeans.c ingest.c normalize.c distance.c centroid.c cluster.c update_centroid.c -o kmeans -lm

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
    Vinho *dados = dataset->dados;

    // aplicar normalizaçao
    normalizarDataset(dataset);

    int tamanhoDataset = dataset->linhas;
    alterarVariavelGlobal(dataset->colunas); // atualizar a variável global com o número de colunas do dataset

    // aplicar o centroides
    Centroide *centroides = inicializarCentroides(dataset, K_CLUSTERS);

    // TESTE CRIACAO DE CENTROIDES //
    printf("\nCentroides iniciais:\n\n");

    for (int i = 0; i < K_CLUSTERS; i++) {
        printf("Centroide %d:\n", i);
        for (int j = 0; j < NUM_FEATURES; j++) {
            printf("%.3f ",
            centroides[i].features[j]);
        }
        printf("\n\n");
    }

    // pra atribuir clusters
    atribuirClusters(dataset, centroides, K_CLUSTERS);

    
    
    printf("\nCentroides apos atualizacao:\n\n");
    // atualizacao dos centroides conforme cluster
    atualizarCentroides(dataset, centroides, K_CLUSTERS);
    for (int i = 0; i < K_CLUSTERS; i++) {

        printf("Centroide %d:\n", i);

        for (int j = 0; j < NUM_FEATURES; j++) {

            printf(
                "%.3f ",
                centroides[i].features[j]
            );
        }

        printf("\n\n");
    }






    printf("\nPrimeiros vinhos agrupados:\n\n");

    for (int i = 0; i < 10; i++) {
        printf("Vinho %d -> Cluster %d\n", i, dataset->dados[i].cluster);
    }






    
    int contagem[K_CLUSTERS] = {0};

    for (int i = 0; i < tamanhoDataset; i++) {
        contagem[dados[i].cluster]++;
    }

    printf("\nDistribuicao dos clusters:\n");

    for (int i = 0; i < K_CLUSTERS; i++) {
        printf("Cluster %d: %d vinhos\n", i, contagem[i]);
    }

    // testar distance.h 
    double distancia = distanciaEuclidiana(dados[0].features, dados[1].features, NUM_FEATURES);

    printf("\nDistancia entre vinho 0 e vinho 1: %f\n", distancia);



    printf("Dados carregados: %d linhas e %d colunas (features).\n", tamanhoDataset, NUM_FEATURES);


    // free
    liberarCentroides(centroides, K_CLUSTERS);
    liberarDataset(dataset);
    return 0;
}
