#ifndef CENTROID_H
#define CENTROID_H

#include "tipos.h"

// criar uma copia dos centroides
Centroide* copiarCentroides(Centroide *originais, int k, int numFeatures);

// pra criar e inicializar centroides aleatorios
Centroide* inicializarCentroides(Dataset *dataset, int k);

// Lliberar a memoria dos centroides
void liberarCentroides(Centroide *centroides, int k);

#endif