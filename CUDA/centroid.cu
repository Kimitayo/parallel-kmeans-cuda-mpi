#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#include "centroid.h"

// ===========================================================================
// centroid.cu
//
// inicializarCentroides, copiarCentroides e liberarCentroides sao
// IDENTICAS ao centroid.c sequencial -- sao chamadas poucas vezes (nao
// estao dentro do loop pesado de iteracoes) e trabalham com pouquissimos
// dados (k centroides), entao nao ha ganho real em colocar na GPU. Mantemos
// elas em CPU de proposito, pra nao complicar o codigo sem necessidade.
//
// Apenas atualizarCentroides foi adaptada para rodar na GPU, pois ela
// percorre TODOS os vinhos a cada iteracao (mesmo papel pesado que tem na
// versao sequencial e na MPI+OpenMP).
// ===========================================================================

#define CUDA_CHECK(chamada)                                                          \
    do {                                                                             \
        cudaError_t erro = (chamada);                                                \
        if (erro != cudaSuccess) {                                                   \
            fprintf(stderr, "Erro CUDA em %s:%d -> %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(erro));                                       \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

// OBS: em CUDA Toolkit 11+/12+, atomicAdd para "double" ja' vem definido
// pelos headers padrao para qualquer arquitetura de destino (internamente
// o compilador escolhe a versao nativa em GPUs CC >= 6.0, ou gera o
// equivalente via atomicCAS em GPUs mais antigas). Se voce estiver usando
// um Toolkit MUITO antigo (CUDA < 8) e der erro de "atomicAdd nao
// encontrado", adicione manualmente a versao por atomicCAS descrita na
// documentacao da NVIDIA (CUDA C Programming Guide, secao B.12).

// --------------------- identicas ao sequencial (CPU) ---------------------

Centroide* inicializarCentroides(Dataset *dataset, int k) {
    int numFeatures = dataset->colunas;

    Centroide *centroides = (Centroide*) malloc(k * sizeof(Centroide));

    if (!centroides) {
        printf("Erro ao alocar centroides\n");
        exit(1);
    }

    // inicializa com seed aleatoria
    // Semente FIXA -- mesmo motivo das outras versoes (ver centroid.c da
    // sequencial): garante centroides iniciais iguais entre TODAS as 4
    // versoes, pra speedup/eficiencia serem comparaveis de verdade.
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
    Centroide *copia = (Centroide*) malloc(k * sizeof(Centroide));

    if (!copia) {
        printf("Erro ao alocar copia dos centroides.\n");
        exit(1);
    }

    for (int i = 0; i < k; i++) {
        copia[i].features = (double*) malloc(numFeatures * sizeof(double));

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

// --------------------- adaptada para GPU ---------------------

// Kernel: cada thread cuida de UM vinho (mesmo papel da iteracao "i" do
// for sequencial) e soma suas features no acumulador do cluster ao qual
// pertence. Usa atomicAdd porque varias threads de clusters diferentes
// podem escrever na mesma posicao de soma/contagem ao mesmo tempo.
__global__ void kernelAcumularCentroides(
    const double *features,
    const int *clusterAtual,
    int linhas,
    int colunas,
    int k,
    double *soma,
    int *contagem
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= linhas) return;

    int cluster = clusterAtual[i];
    if (cluster < 0 || cluster >= k) return;

    atomicAdd(&contagem[cluster], 1);

    for (int f = 0; f < colunas; f++) {
        atomicAdd(&soma[(size_t) cluster * colunas + f], features[(size_t) i * colunas + f]);
    }
}

void atualizarCentroides(Dataset *dataset, Centroide *centroides, int k) {
    int linhas = dataset->linhas;
    int colunas = dataset->colunas;

    // ---- achatar dataset (features + cluster atual) ----
    double *h_features = (double *) malloc((size_t) linhas * colunas * sizeof(double));
    int *h_cluster = (int *) malloc((size_t) linhas * sizeof(int));

    for (int i = 0; i < linhas; i++) {
        memcpy(h_features + (size_t) i * colunas, dataset->dados[i].features, colunas * sizeof(double));
        h_cluster[i] = dataset->dados[i].cluster;
    }

    // ---- copiar para a GPU e zerar acumuladores ----
    double *d_features, *d_soma;
    int *d_cluster, *d_contagem;

    CUDA_CHECK(cudaMalloc(&d_features, (size_t) linhas * colunas * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cluster, (size_t) linhas * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_soma, (size_t) k * colunas * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_contagem, (size_t) k * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_features, h_features, (size_t) linhas * colunas * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cluster, h_cluster, (size_t) linhas * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_soma, 0, (size_t) k * colunas * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_contagem, 0, (size_t) k * sizeof(int)));

    // ---- rodar o kernel (equivalente ao "for" do sequencial) ----
    int threadsPorBloco = 256;
    int blocos = (linhas + threadsPorBloco - 1) / threadsPorBloco;

    kernelAcumularCentroides<<<blocos, threadsPorBloco>>>(
        d_features, d_cluster, linhas, colunas, k, d_soma, d_contagem);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- trazer soma/contagem de volta (e' pequeno: k*colunas) ----
    double *h_soma = (double *) malloc((size_t) k * colunas * sizeof(double));
    int *h_contagem = (int *) malloc((size_t) k * sizeof(int));

    CUDA_CHECK(cudaMemcpy(h_soma, d_soma, (size_t) k * colunas * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_contagem, d_contagem, (size_t) k * sizeof(int), cudaMemcpyDeviceToHost));

    // ---- calcular a media; clusters vazios preservam o centroide anterior
    //      (sem zerar antes -- um centroide zerado, com dados normalizados
    //      entre 0 e 1, pode atrair pontos artificialmente pra um canto do
    //      espaco de features que nao tem nada a ver com os dados reais) ----
    for (int c = 0; c < k; c++) {
        if (h_contagem[c] == 0) continue;

        for (int f = 0; f < colunas; f++) {
            centroides[c].features[f] = h_soma[c * colunas + f] / h_contagem[c];
        }
    }

    free(h_features);
    free(h_cluster);
    free(h_soma);
    free(h_contagem);
    cudaFree(d_features);
    cudaFree(d_cluster);
    cudaFree(d_soma);
    cudaFree(d_contagem);
}
