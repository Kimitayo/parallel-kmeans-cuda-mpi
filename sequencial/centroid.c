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
    // Semente FIXA (nao time(NULL)) --> importante para benchmarking:
    // assim todas as execucoes (sequencial, MPI+OpenMP, CUDA, OpenMP-GPU)
    // partem dos MESMOS centroides iniciais, e o numero de iteracoes ate
    // convergir fica igual entre elas. Sem isso, a comparacao de tempo de
    // execucao mistura "sorte do sorteio aleatorio" com "ganho real do
    // paralelismo", o que invalida o calculo de speedup/eficiencia.
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

    // quantidade de vinhos por cluster
    int *contagem = (int*) calloc(k, sizeof(int));

    double **soma = malloc(k * sizeof(double*));

    for (int c = 0; c < k; c++) {
        soma[c] = calloc(numFeatures, sizeof(double));
    }

    if (!contagem) {
        printf("Erro ao alocar contagem.\n");
        exit(1);
    }

    // soma features dos vinhos
    for (int i = 0; i < dataset->linhas; i++) {
        int cluster = dataset->dados[i].cluster;
        contagem[cluster]++;

        for (int f = 0; f < numFeatures; f++) {
            soma[cluster][f] += dataset->dados[i].features[f];
        }
    }

    // divide pela quantidade. clusters vazios preservam o centroide anterior
    // (sem zerar antes => um centroide zerado, com dados normalizados entre
    // 0 e 1, pode atrair pontos artificialmente pra um canto do espaco de
    // features que nao tem nada a ver com a distribuicao real dos dados)
    for (int c = 0; c < k; c++) {
        if (contagem[c] == 0)
            continue;

        for (int f = 0; f < numFeatures; f++) {
            centroides[c].features[f] = soma[c][f] / contagem[c];
        }
    }

    free(contagem);
    for (int c = 0; c < k; c++) {
        free(soma[c]);
    }
    free(soma);
}

void liberarCentroides(Centroide *centroides, int k) {
    for (int i = 0; i < k; i++) {
        free(centroides[i].features);
    }

    free(centroides);
}