#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "tipos.h"
#include "ingest.h"
#include "normalize.h"
#include "centroid.h"
#include "cluster.h"
#include "convergence.h"
#include "sse.h"

// gcc -fopenmp kmeans.c ingest.c normalize.c distance.c centroid.c cluster.c convergence.c sse.c -o kmeans -lm

int K_CLUSTERS = 3; // Quantidade de grupos que queremos encontrar
int MAX_ITER_ACOES = 100; // Critério de parada por repetições
int NUM_FEATURES = 11; // quantidade de colunas (exceto id e qualidade)

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


void executarKmeans(Dataset *dataset, Centroide *centroides) {
    
    // pra convergencia
    for (int iter = 0; iter < MAX_ITER_ACOES; iter++) {

        // guardar os centroides antigos
        Centroide *antigos = copiarCentroides(centroides, K_CLUSTERS, NUM_FEATURES);

        // colocar cluster
        int mudouCluster = atribuirClusters(dataset, centroides, K_CLUSTERS);
        // atualizar centroide
        atualizarCentroides(dataset, centroides, K_CLUSTERS);

        // verificar convergencia
        if (!mudouCluster || convergiu(antigos, centroides, K_CLUSTERS, NUM_FEATURES, 0.0001)) {
            printf("\nConvergiu na iteracao %d\n", iter + 1);
            liberarCentroides(antigos, K_CLUSTERS);
            break;
        }

        liberarCentroides(antigos, K_CLUSTERS);
    }

}


int main(int argc, char *argv[]) {
    const char *nomeArquivo = "vinhos.csv";
    
    if (argc == 2) {
        K_CLUSTERS = atoi(argv[1]); // atualizar a variável global com a quantidade de clusters desejada
    }else if (argc == 3) {
        K_CLUSTERS = atoi(argv[1]);
        MAX_ITER_ACOES = atoi(argv[2]); // atualizar a variável global com a quantidade de iterações máximas desejada para o algoritmo
    }

    // medir tempo inicial
    double inicio = omp_get_wtime();

    Dataset *dataset = carregarDados(nomeArquivo);

    // atualizar a variável global com o número de colunas extraídas do dataset
    // não conta coluna de id nem qualidade
    NUM_FEATURES = dataset->colunas; 

    Vinho *dados = dataset->dados;

    // aplicar normalizaçao
    normalizarDataset(dataset);

    // inicializar centroide
    Centroide *centroides = inicializarCentroides(dataset, K_CLUSTERS);

    executarKmeans(dataset, centroides);

    // tempo final
    double fim = omp_get_wtime();
    printf("\nTempo de execucao: %.6f segundos\n", fim - inicio);

    // mostrar os centroides finais
    printf("\nCentroides finais:\n\n");
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

    // mostrar alguns vinhos
    printf("\nPrimeiros vinhos agrupados:\n\n");

    for (int i = 0; i < 10; i++) {
        printf("Vinho %d -> Cluster %d\n", i, dataset->dados[i].cluster);
    }

    // contagem por cluster
    int contagem[K_CLUSTERS];

    memset(contagem, 0, sizeof(contagem));

    for (int i = 0; i < dataset->linhas; i++) {
        contagem[dados[i].cluster]++;
    }

    printf("\nDistribuicao dos clusters:\n");

    for (int i = 0; i < K_CLUSTERS; i++) {
        printf("Cluster %d: %d vinhos\n", i, contagem[i]);
    }


    // verificar a qualidade do agrupamento
    double sse = calcularSSE(dataset, centroides);
    printf("\nQualidade do agrupamento (SSE): %.6f\n", sse);
    double sseMedio = sse / dataset->linhas;
    printf("SSE medio: %.6f\n", sseMedio);

    // free
    liberarCentroides(centroides, K_CLUSTERS);
    liberarDataset(dataset);
    return 0;
}
