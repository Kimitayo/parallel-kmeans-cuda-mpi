#include <math.h>
#include <mpi.h>
#include <omp.h>

#include "sse.h"
#include "distance.h"

double calcularSSE(Dataset *dataset, Centroide *centroides, MPI_Comm comm) {
    double sseLocal = 0.0;
    double sseGlobal = 0.0;

    #pragma omp parallel for reduction(+:sseLocal)
    for (int i = 0; i < dataset->linhas; i++) {
        int cluster = dataset->dados[i].cluster;

        if (cluster >= 0) {
            // distanciaEuclidiana() ja' retorna ao quadrado --> nao multiplicar de novo
            double distanciaAoQuadrado = distanciaEuclidiana(dataset->dados[i].features, centroides[cluster].features, dataset->colunas);
            sseLocal += distanciaAoQuadrado;
        }
    }

    MPI_Allreduce(&sseLocal, &sseGlobal, 1, MPI_DOUBLE, MPI_SUM, comm);

    return sseGlobal;
}
