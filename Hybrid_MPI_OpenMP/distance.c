#include <math.h>
#include <omp.h>
#include "distance.h"

double distanciaEuclidiana(
    double *ponto1,
    double *ponto2,
    int numFeatures
) {
    // OBS: retorna a distancia EUCLIDIANA AO QUADRADO (sem sqrt) de
    // proposito -- mesmo motivo da versao sequencial (ver comentario lá):
    // evita um sqrt() extra por par (ponto, centroide) a cada iteracao,
    // sem mudar o resultado de "qual centroide esta mais perto".

    double soma = 0.0;

    #pragma omp simd reduction(+:soma)
    for (int i = 0; i < numFeatures; i++) {

        double diferenca = ponto1[i] - ponto2[i];
        soma += diferenca * diferenca;
    }

    return soma;
}
