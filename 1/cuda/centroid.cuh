#ifndef CENTROID_H
#define CENTROID_H

void alocarCentroides(double **centroides, double **centroidesAntigos, double **somaCentroides, int **contagemClusters, int k, int numFeatures);

void inicializarCentroides(const double *features, double *centroides, int numSamples, int numFeatures, int k);

void copiarCentroides(const double *origem, double *destino, int k, int numFeatures);

void atualizarCentroides(const double *features, const int *clusters, double *centroides, double *somaCentroides, int *contagemClusters, int numSamples, int numFeatures, int k);

void liberarCentroides(double *centroides, double *centroidesAntigos, double *somaCentroides, int *contagemClusters);

#endif