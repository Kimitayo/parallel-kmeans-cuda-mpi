#ifndef DISTANCE_H
#define DISTANCE_H

#pragma omp declare target
double distanciaEuclidiana(
    double *ponto1,
    double *ponto2,
    int numFeatures
);
#pragma omp end declare target

#endif