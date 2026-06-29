#include <math.h>

#include "sse.h"
#include "distance.h"

double calcularSSE(Dataset *dataset, Centroide *centroides) {
    double sse = 0.0;

    for (int i = 0; i < dataset->linhas; i++) {
        int cluster = dataset->dados[i].cluster;


        double distanciaAoQuadrado = distanciaEuclidiana(dataset->dados[i].features, centroides[cluster].features, dataset->colunas);
        sse += distanciaAoQuadrado;
    }

    return sse;
}