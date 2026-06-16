#include <math.h>
#include <omp.h>

#include "sse.h"
#include "distance.h"

double calcularSSE(Dataset *dataset, Centroide *centroides) {
    double sse = 0.0;
    
    #pragma omp parallel for reduction(+:sse)
    for (int i = 0; i < dataset->linhas; i++) {
        int cluster = dataset->dados[i].cluster;

        double distancia = distanciaEuclidiana(dataset->dados[i].features, centroides[cluster].features, dataset->colunas);
        sse += distancia * distancia;
    }

    return sse;
}