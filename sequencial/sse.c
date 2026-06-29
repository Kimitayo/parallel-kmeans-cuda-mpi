#include <math.h>

#include "sse.h"
#include "distance.h"

double calcularSSE(Dataset *dataset, Centroide *centroides) {
    double sse = 0.0;

    for (int i = 0; i < dataset->linhas; i++) {
        int cluster = dataset->dados[i].cluster;

        // distanciaEuclidiana() ja' retorna ao quadrado -- nao multiplicar
        // de novo (senao o resultado vira distancia^4, nao SSE de verdade)
        double distanciaAoQuadrado = distanciaEuclidiana(dataset->dados[i].features, centroides[cluster].features, dataset->colunas);
        sse += distanciaAoQuadrado;
    }

    return sse;
}