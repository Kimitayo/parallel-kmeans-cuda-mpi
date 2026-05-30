#include <float.h>
#include "cluster.h"
#include "distance.h"

void atribuirClusters(Dataset *dataset, Centroide *centroides, int k) {
    for (int i = 0; i < dataset->linhas; i++) {
        double menorDistancia = DBL_MAX;
        int melhorCluster = -1;

        for (int c = 0; c < k; c++) {
            double distancia =
                distanciaEuclidiana(dataset->dados[i].features, centroides[c].features, dataset->colunas);

            if (distancia < menorDistancia) {
                menorDistancia = distancia;
                melhorCluster = c;
            }
        }
        dataset->dados[i].cluster = melhorCluster;
    }
}