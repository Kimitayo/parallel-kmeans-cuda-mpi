#include <float.h>
#include "cluster.h"
#include "distance.h"

// retorna 1 se algum ponto mudou de cluster, 0 caso contrário
int atribuirClusters(Dataset *dataset, Centroide *centroides, int k) {
    int mudouCluster = 0;

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

        if (dataset->dados[i].cluster != melhorCluster) {
            mudouCluster = 1;
        }

        dataset->dados[i].cluster = melhorCluster;
    }
    return mudouCluster;
}