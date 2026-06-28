#include <math.h>
#include "distance.cuh"

__host__ __device__ double distanciaEuclidiana(const double *ponto1, const double *ponto2, int numFeatures) {

    double soma = 0.0;

    for (int i = 0; i < numFeatures; i++) {

        double diferenca = ponto1[i] - ponto2[i];
        soma += diferenca * diferenca;
    }

    return soma;
}