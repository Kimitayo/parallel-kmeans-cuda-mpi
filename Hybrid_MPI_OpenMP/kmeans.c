#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#include "tipos.h"
#include "ingest.h"
#include "normalize.h"
#include "centroid.h"
#include "cluster.h"
#include "convergence.h"
#include "sse.h"

// mpicc -fopenmp kmeans.c ingest.c normalize.c distance.c centroid.c cluster.c convergence.c sse.c -o kmeans -lm

int K_CLUSTERS = 3;
int MAX_ITER_ACOES = 100;
int NUM_FEATURES = 11;

void liberarDataset(Dataset *dataset) {
    if (dataset) {
        if (dataset->dados) {
            for (int i = 0; i < dataset->linhas; i++) {
                free(dataset->dados[i].features);
            }

            free(dataset->dados);
        }

        free(dataset);
    }
}

static void abortarMPI(MPI_Comm comm, const char *mensagem) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0 && mensagem) {
        printf("%s\n", mensagem);
    }

    MPI_Abort(comm, 1);
}

static Dataset* criarDatasetLocal(int linhas, int colunas, MPI_Comm comm) {
    Dataset *dataset = malloc(sizeof(Dataset));
    Vinho *dados = calloc((size_t) (linhas > 0 ? linhas : 1), sizeof(Vinho));

    if (!dataset || !dados) {
        free(dataset);
        free(dados);
        abortarMPI(comm, "Erro ao alocar dataset local.");
    }

    dataset->dados = dados;
    dataset->linhas = linhas;
    dataset->colunas = colunas;

    for (int i = 0; i < linhas; i++) {
        dataset->dados[i].features = malloc((size_t) colunas * sizeof(double));

        if (!dataset->dados[i].features) {
            liberarDataset(dataset);
            abortarMPI(comm, "Erro ao alocar features do dataset local.");
        }

        dataset->dados[i].cluster = -1;
    }

    return dataset;
}

static double* serializarFeatures(Dataset *dataset, MPI_Comm comm) {
    size_t total = (size_t) dataset->linhas * (size_t) dataset->colunas;
    double *features = malloc((total > 0 ? total : 1) * sizeof(double));

    if (!features) {
        abortarMPI(comm, "Erro ao serializar features.");
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dataset->linhas; i++) {
        for (int j = 0; j < dataset->colunas; j++) {
            features[(size_t) i * (size_t) dataset->colunas + j] = dataset->dados[i].features[j];
        }
    }

    return features;
}

static void preencherDatasetLocal(
    Dataset *dataset,
    double *features,
    int *qualidades,
    int *ids
) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < dataset->linhas; i++) {
        for (int j = 0; j < dataset->colunas; j++) {
            dataset->dados[i].features[j] = features[(size_t) i * (size_t) dataset->colunas + j];
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dataset->linhas; i++) {
        dataset->dados[i].qualidade = qualidades[i];
        dataset->dados[i].id = ids[i];
        dataset->dados[i].cluster = -1;
    }
}

static Centroide* alocarCentroides(int k, int numFeatures, MPI_Comm comm) {
    Centroide *centroides = calloc((size_t) k, sizeof(Centroide));

    if (!centroides) {
        abortarMPI(comm, "Erro ao alocar centroides.");
    }

    for (int i = 0; i < k; i++) {
        centroides[i].features = calloc((size_t) numFeatures, sizeof(double));

        if (!centroides[i].features) {
            liberarCentroides(centroides, k);
            abortarMPI(comm, "Erro ao alocar features dos centroides.");
        }
    }

    return centroides;
}

static void calcularDistribuicao(
    int totalLinhas,
    int colunas,
    int size,
    int *linhasPorRank,
    int *deslocLinhas,
    int *featuresPorRank,
    int *deslocFeatures
) {
    int base = totalLinhas / size;
    int resto = totalLinhas % size;
    int desloc = 0;

    for (int r = 0; r < size; r++) {
        linhasPorRank[r] = base + (r < resto ? 1 : 0);
        deslocLinhas[r] = desloc;
        featuresPorRank[r] = linhasPorRank[r] * colunas;
        deslocFeatures[r] = desloc * colunas;
        desloc += linhasPorRank[r];
    }
}

static void preencherMetadados(Dataset *dataset, int *qualidades, int *ids) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dataset->linhas; i++) {
        qualidades[i] = dataset->dados[i].qualidade;
        ids[i] = dataset->dados[i].id;
    }
}

static void copiarBufferParaCentroides(double *buffer, Centroide *centroides, int k, int numFeatures) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int c = 0; c < k; c++) {
        for (int f = 0; f < numFeatures; f++) {
            centroides[c].features[f] = buffer[c * numFeatures + f];
        }
    }
}

void executarKmeans(Dataset *dataset, Centroide *centroides, int rank, MPI_Comm comm) {
    for (int iter = 0; iter < MAX_ITER_ACOES; iter++) {
        Centroide *antigos = copiarCentroides(centroides, K_CLUSTERS, NUM_FEATURES);

        int mudouLocal = atribuirClusters(dataset, centroides, K_CLUSTERS);
        int mudouGlobal = 0;
        MPI_Allreduce(&mudouLocal, &mudouGlobal, 1, MPI_INT, MPI_MAX, comm);

        atualizarCentroides(dataset, centroides, K_CLUSTERS, comm);

        int convergiuLocal = convergiu(antigos, centroides, K_CLUSTERS, NUM_FEATURES, 0.0001);
        int convergiuGlobal = 0;
        MPI_Allreduce(&convergiuLocal, &convergiuGlobal, 1, MPI_INT, MPI_MIN, comm);

        liberarCentroides(antigos, K_CLUSTERS);

        if (!mudouGlobal || convergiuGlobal) {
            if (rank == 0) {
                printf("\nConvergiu na iteracao %d\n", iter + 1);
            }
            break;
        }
    }
}

int main(int argc, char *argv[]) {
    const char *nomeArquivo = "../vinhos.csv";

    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc == 2) {
        K_CLUSTERS = atoi(argv[1]);
    } else if (argc == 3) {
        K_CLUSTERS = atoi(argv[1]);
        MAX_ITER_ACOES = atoi(argv[2]);
    }

    double inicio = MPI_Wtime();

    Dataset *datasetCompleto = NULL;
    int totalLinhas = 0;
    int totalColunas = 0;

    if (rank == 0) {
        datasetCompleto = carregarDados(nomeArquivo);
        totalLinhas = datasetCompleto->linhas;
        totalColunas = datasetCompleto->colunas;
    }

    MPI_Bcast(&totalLinhas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&totalColunas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K_CLUSTERS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MAX_ITER_ACOES, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (totalLinhas <= 0 || totalColunas <= 0 || K_CLUSTERS <= 0) {
        abortarMPI(MPI_COMM_WORLD, "Parametros ou dataset invalidos.");
    }

    NUM_FEATURES = totalColunas;

    int *linhasPorRank = malloc((size_t) size * sizeof(int));
    int *deslocLinhas = malloc((size_t) size * sizeof(int));
    int *featuresPorRank = malloc((size_t) size * sizeof(int));
    int *deslocFeatures = malloc((size_t) size * sizeof(int));

    if (!linhasPorRank || !deslocLinhas || !featuresPorRank || !deslocFeatures) {
        abortarMPI(MPI_COMM_WORLD, "Erro ao alocar distribuicao MPI.");
    }

    calcularDistribuicao(totalLinhas, totalColunas, size, linhasPorRank, deslocLinhas, featuresPorRank, deslocFeatures);

    int linhasLocais = linhasPorRank[rank];
    int featuresLocais = featuresPorRank[rank];

    double *featuresSerializadas = NULL;
    int *qualidades = NULL;
    int *ids = NULL;

    if (rank == 0) {
        featuresSerializadas = serializarFeatures(datasetCompleto, MPI_COMM_WORLD);
        qualidades = malloc((size_t) totalLinhas * sizeof(int));
        ids = malloc((size_t) totalLinhas * sizeof(int));

        if (!qualidades || !ids) {
            abortarMPI(MPI_COMM_WORLD, "Erro ao alocar metadados do dataset.");
        }

        preencherMetadados(datasetCompleto, qualidades, ids);
    }

    double *featuresRecebidas = malloc((size_t) (featuresLocais > 0 ? featuresLocais : 1) * sizeof(double));
    int *qualidadesRecebidas = malloc((size_t) (linhasLocais > 0 ? linhasLocais : 1) * sizeof(int));
    int *idsRecebidos = malloc((size_t) (linhasLocais > 0 ? linhasLocais : 1) * sizeof(int));

    if (!featuresRecebidas || !qualidadesRecebidas || !idsRecebidos) {
        abortarMPI(MPI_COMM_WORLD, "Erro ao alocar buffers locais.");
    }

    MPI_Scatterv(featuresSerializadas, featuresPorRank, deslocFeatures, MPI_DOUBLE,
                 featuresRecebidas, featuresLocais, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(qualidades, linhasPorRank, deslocLinhas, MPI_INT,
                 qualidadesRecebidas, linhasLocais, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(ids, linhasPorRank, deslocLinhas, MPI_INT,
                 idsRecebidos, linhasLocais, MPI_INT, 0, MPI_COMM_WORLD);

    Dataset *dataset = criarDatasetLocal(linhasLocais, totalColunas, MPI_COMM_WORLD);
    preencherDatasetLocal(dataset, featuresRecebidas, qualidadesRecebidas, idsRecebidos);

    free(featuresRecebidas);
    free(qualidadesRecebidas);
    free(idsRecebidos);
    free(featuresSerializadas);
    free(qualidades);
    free(ids);

    if (rank == 0) {
        liberarDataset(datasetCompleto);
    }

    normalizarDataset(dataset, MPI_COMM_WORLD);

    Centroide *centroides = alocarCentroides(K_CLUSTERS, NUM_FEATURES, MPI_COMM_WORLD);
    double *bufferCentroides = malloc((size_t) K_CLUSTERS * (size_t) NUM_FEATURES * sizeof(double));

    if (!bufferCentroides) {
        abortarMPI(MPI_COMM_WORLD, "Erro ao alocar buffer dos centroides.");
    }

    double *featuresNormalizadasLocais = serializarFeatures(dataset, MPI_COMM_WORLD);
    double *featuresNormalizadasGlobais = NULL;

    if (rank == 0) {
        featuresNormalizadasGlobais = malloc((size_t) totalLinhas * (size_t) NUM_FEATURES * sizeof(double));

        if (!featuresNormalizadasGlobais) {
            abortarMPI(MPI_COMM_WORLD, "Erro ao reunir features normalizadas.");
        }
    }

    MPI_Gatherv(featuresNormalizadasLocais, featuresLocais, MPI_DOUBLE,
                featuresNormalizadasGlobais, featuresPorRank, deslocFeatures, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // semente fixa
        srand(42);

        for (int c = 0; c < K_CLUSTERS; c++) {
            int indiceAleatorio = rand() % totalLinhas;

            for (int f = 0; f < NUM_FEATURES; f++) {
                bufferCentroides[c * NUM_FEATURES + f] =
                    featuresNormalizadasGlobais[(size_t) indiceAleatorio * (size_t) NUM_FEATURES + f];
            }
        }
    }

    free(featuresNormalizadasLocais);
    free(featuresNormalizadasGlobais);

    MPI_Bcast(bufferCentroides, K_CLUSTERS * NUM_FEATURES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    copiarBufferParaCentroides(bufferCentroides, centroides, K_CLUSTERS, NUM_FEATURES);
    free(bufferCentroides);

    executarKmeans(dataset, centroides, rank, MPI_COMM_WORLD);

    double fim = MPI_Wtime();

    if (rank == 0) {
        printf("\nTempo de execucao: %.6f segundos\n", fim - inicio);
        printf("\nCentroides finais:\n\n");

        for (int i = 0; i < K_CLUSTERS; i++) {
            printf("Centroide %d:\n", i);

            for (int j = 0; j < NUM_FEATURES; j++) {
                printf("%.3f ", centroides[i].features[j]);
            }

            printf("\n\n");
        }
    }

    int *clustersLocais = malloc((size_t) (linhasLocais > 0 ? linhasLocais : 1) * sizeof(int));

    if (!clustersLocais) {
        abortarMPI(MPI_COMM_WORLD, "Erro ao alocar clusters locais.");
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < linhasLocais; i++) {
        clustersLocais[i] = dataset->dados[i].cluster;
    }

    int *clustersGlobais = NULL;
    if (rank == 0) {
        clustersGlobais = malloc((size_t) totalLinhas * sizeof(int));

        if (!clustersGlobais) {
            abortarMPI(MPI_COMM_WORLD, "Erro ao alocar clusters globais.");
        }
    }

    MPI_Gatherv(clustersLocais, linhasLocais, MPI_INT,
                clustersGlobais, linhasPorRank, deslocLinhas, MPI_INT,
                0, MPI_COMM_WORLD);

    int *contagemLocal = calloc((size_t) K_CLUSTERS, sizeof(int));
    int *contagemGlobal = calloc((size_t) K_CLUSTERS, sizeof(int));
    int numThreads = omp_get_max_threads();
    int *contagemThreads = calloc((size_t) numThreads * (size_t) K_CLUSTERS, sizeof(int));

    if (!contagemLocal || !contagemGlobal || !contagemThreads) {
        abortarMPI(MPI_COMM_WORLD, "Erro ao alocar contagem dos clusters.");
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int *contagemThread = contagemThreads + ((size_t) tid * (size_t) K_CLUSTERS);

        #pragma omp for schedule(static)
        for (int i = 0; i < dataset->linhas; i++) {
            int cluster = dataset->dados[i].cluster;

            if (cluster >= 0 && cluster < K_CLUSTERS) {
                contagemThread[cluster]++;
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < K_CLUSTERS; c++) {
        int total = 0;

        for (int t = 0; t < numThreads; t++) {
            total += contagemThreads[(size_t) t * (size_t) K_CLUSTERS + c];
        }

        contagemLocal[c] = total;
    }

    MPI_Reduce(contagemLocal, contagemGlobal, K_CLUSTERS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nPrimeiros vinhos agrupados:\n\n");

        int limite = totalLinhas < 10 ? totalLinhas : 10;
        for (int i = 0; i < limite; i++) {
            printf("Vinho %d -> Cluster %d\n", i, clustersGlobais[i]);
        }

        printf("\nDistribuicao dos clusters:\n");

        for (int i = 0; i < K_CLUSTERS; i++) {
            printf("Cluster %d: %d vinhos\n", i, contagemGlobal[i]);
        }
    }

    double sse = calcularSSE(dataset, centroides, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nQualidade do agrupamento (SSE): %.6f\n", sse);
        printf("SSE medio: %.6f\n", sse / totalLinhas);
    }

    free(clustersLocais);
    free(clustersGlobais);
    free(contagemLocal);
    free(contagemGlobal);
    free(contagemThreads);
    free(linhasPorRank);
    free(deslocLinhas);
    free(featuresPorRank);
    free(deslocFeatures);

    liberarCentroides(centroides, K_CLUSTERS);
    liberarDataset(dataset);

    MPI_Finalize();
    return 0;
}
