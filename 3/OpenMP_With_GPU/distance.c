#include <math.h>
#include "distance.h"

#pragma omp declare target
double distanciaEuclidiana(
    double *ponto1,
    double *ponto2,
    int numFeatures
) {

    double soma = 0.0;

    for (int i = 0; i < numFeatures; i++) {

        double diferenca = ponto1[i] - ponto2[i];
        soma += diferenca * diferenca;
    }

    return soma;
}
#pragma omp end declare target