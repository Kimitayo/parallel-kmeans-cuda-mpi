#include <math.h>
#include <omp.h>
#include "distance.h"

double distanciaEuclidiana(
    double *ponto1,
    double *ponto2,
    int numFeatures
) {

    double soma = 0.0;

    #pragma omp simd reduction(+:soma)
    for (int i = 0; i < numFeatures; i++) {

        double diferenca = ponto1[i] - ponto2[i];
        soma += diferenca * diferenca;
    }

    return soma;
}
