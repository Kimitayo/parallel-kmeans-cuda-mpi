#include <math.h>

#include "sse.h"
#include "distance.h"

double calcularSSE(Dataset *dataset, Centroide *centroides) {
    double sse = 0.0;

    for (int i = 0; i < dataset->linhas; i++) {
        int cluster = dataset->dados[i].cluster;

        // distanciaEuclidiana() retorna a distancia AO QUADRADO (sem sqrt,
        // de proposito  Como SSE e
        // "soma das distancias ao quadrado", basta somar direto, sem
        // multiplicar de novo (senao o resultado vira distancia^4).
        double distanciaAoQuadrado = distanciaEuclidiana(dataset->dados[i].features, centroides[cluster].features, dataset->colunas);
        sse += distanciaAoQuadrado;
    }

    return sse;
}