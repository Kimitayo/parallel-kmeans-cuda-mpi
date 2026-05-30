#ifndef CENTROID_H
#define CENTROID_H

#include "tipos.h"

// pra criar e inicializar centroides aleatorios
Centroide* inicializarCentroides(Dataset *dataset, int k);

// Lliberar a memoria dos centroides
void liberarCentroides(Centroide *centroides, int k);

#endif