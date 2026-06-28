#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#include "tipos.h"
#include "normalize.h"
#include "centroid.h"
#include "cluster.h"

// ingest.h, convergence.h e sse.h sao implementados em arquivos .c puro
// (compilados como C, nao C++). Sem o extern "C" abaixo, o nvcc (que trata
// .cu como C++) geraria nomes "mangled" diferentes dos simbolos reais
// gerados pelo gcc/host compiler para esses .c, e o link falharia.
extern "C" {
#include "ingest.h"
#include "convergence.h"
#include "sse.h"
}

#define CUDA_CHECK(chamada)                                                          \
    do {                                                                             \
        cudaError_t erro = (chamada);                                                \
        if (erro != cudaSuccess) {                                                   \
            fprintf(stderr, "Erro CUDA em %s:%d -> %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(erro));                                       \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

// nvcc kmeans.cu ingest.c normalize.cu distance.c centroid.cu cluster.cu convergence.c sse.c -o kmeans_cuda -lm


int K_CLUSTERS = 3;       // Quantidade de grupos que queremos encontrar
int MAX_ITER_ACOES = 100; // Critério de parada por repetições
int NUM_FEATURES = 11;    // quantidade de colunas (exceto id e qualidade)

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

        // colocar cluster (roda na GPU por dentro, ver cluster.cu)
        int mudouCluster = atribuirClusters(dataset, centroides, K_CLUSTERS);
        // atualizar centroide (roda na GPU por dentro, ver centroid.cu)
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
    const char *nomeArquivo = "../vinhos.csv";

    if (argc == 2) {
        K_CLUSTERS = atoi(argv[1]);
    } else if (argc == 3) {
        K_CLUSTERS = atoi(argv[1]);
        MAX_ITER_ACOES = atoi(argv[2]);
    }

    // identificar a GPU usada (util pra colocar no relatorio/slides)
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Versao: CUDA (GPU) | Dispositivo: %s | Compute Capability: %d.%d\n",
           prop.name, prop.major, prop.minor);

    // medir tempo inicial
    struct timespec inicio, fim;
    clock_gettime(CLOCK_MONOTONIC, &inicio);

    Dataset *dataset = carregarDados(nomeArquivo);

    // atualizar a variável global com o número de colunas extraídas do dataset
    NUM_FEATURES = dataset->colunas;

    Vinho *dados = dataset->dados;

    // aplicar normalizaçao (roda na GPU por dentro, ver normalize.cu)
    normalizarDataset(dataset);

    // inicializar centroide (CPU, igual ao sequencial -- ver centroid.cu)
    Centroide *centroides = inicializarCentroides(dataset, K_CLUSTERS);

    executarKmeans(dataset, centroides);

    // tempo final
    clock_gettime(CLOCK_MONOTONIC, &fim);
    double tempoSegundos = (fim.tv_sec - inicio.tv_sec) + (fim.tv_nsec - inicio.tv_nsec) / 1e9;
    printf("\nTempo de execucao: %.6f segundos\n", tempoSegundos);

    // mostrar os centroides finais
    printf("\nCentroides finais:\n\n");
    for (int i = 0; i < K_CLUSTERS; i++) {
        printf("Centroide %d:\n", i);

        for (int j = 0; j < NUM_FEATURES; j++) {
            printf("%.3f ", centroides[i].features[j]);
        }

        printf("\n\n");
    }

    // mostrar alguns vinhos
    printf("\nPrimeiros vinhos agrupados:\n\n");

    int limite = dataset->linhas < 10 ? dataset->linhas : 10;
    for (int i = 0; i < limite; i++) {
        printf("Vinho %d -> Cluster %d\n", i, dataset->dados[i].cluster);
    }

    // contagem por cluster
    int *contagem = (int *) calloc(K_CLUSTERS, sizeof(int));

    for (int i = 0; i < dataset->linhas; i++) {
        contagem[dados[i].cluster]++;
    }

    printf("\nDistribuicao dos clusters:\n");
    for (int i = 0; i < K_CLUSTERS; i++) {
        printf("Cluster %d: %d vinhos\n", i, contagem[i]);
    }

    // verificar a qualidade do agrupamento (CPU, igual ao sequencial --> ver sse.c)
    double sse = calcularSSE(dataset, centroides);
    printf("\nQualidade do agrupamento (SSE): %.6f\n", sse);
    double sseMedio = sse / dataset->linhas;
    printf("SSE medio: %.6f\n", sseMedio);

    // free
    free(contagem);
    liberarCentroides(centroides, K_CLUSTERS);
    liberarDataset(dataset);
    return 0;
}
