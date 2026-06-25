#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "normalize.h"

void normalizarDataset(Dataset *dataset, MPI_Comm comm) {
    int linhas = dataset->linhas;
    int colunas = dataset->colunas;
    int rank;

    MPI_Comm_rank(comm, &rank);

    double *minLocal = malloc((size_t) colunas * sizeof(double));
    double *maxLocal = malloc((size_t) colunas * sizeof(double));
    double *minGlobal = malloc((size_t) colunas * sizeof(double));
    double *maxGlobal = malloc((size_t) colunas * sizeof(double));

    if (!minLocal || !maxLocal || !minGlobal || !maxGlobal) {
        printf("Erro ao alocar vetores de normalizacao.\n");
        free(minLocal);
        free(maxLocal);
        free(minGlobal);
        free(maxGlobal);
        MPI_Abort(comm, 1);
    }

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < colunas; j++) {
        double menor = DBL_MAX;
        double maior = -DBL_MAX;

        for (int i = 0; i < linhas; i++) {
            double valor = dataset->dados[i].features[j];

            if (valor < menor) menor = valor;
            if (valor > maior) maior = valor;
        }

        minLocal[j] = menor;
        maxLocal[j] = maior;
    }

    MPI_Allreduce(minLocal, minGlobal, colunas, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(maxLocal, maxGlobal, colunas, MPI_DOUBLE, MPI_MAX, comm);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < linhas; i++) {
        for (int j = 0; j < colunas; j++) {
            double valor = dataset->dados[i].features[j];
            double amplitude = maxGlobal[j] - minGlobal[j];

            if (amplitude > 0) {
                dataset->dados[i].features[j] = (valor - minGlobal[j]) / amplitude;
            } else {
                dataset->dados[i].features[j] = 0.0;
            }
        }
    }

    free(minLocal);
    free(maxLocal);
    free(minGlobal);
    free(maxGlobal);

    if (rank == 0) {
        printf("Dataset normalizado com sucesso.\n");
    }
}
