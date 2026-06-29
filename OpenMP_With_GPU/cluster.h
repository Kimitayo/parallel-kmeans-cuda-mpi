#ifndef CLUSTER_H
#define CLUSTER_H

#include "tipos.h"

// retorna 1 se algum ponto mudou de cluster, 0 caso contrário
int atribuirClusters(Dataset *dataset, Centroide *centroides, int k);

#endif