#include <math.h>
#include "distance.h"

double distanciaEuclidiana(
    double *ponto1,
    double *ponto2,
    int numFeatures
) {
    // OBS: retorna a distancia EUCLIDIANA AO QUADRADO (sem sqrt) de
    // proposito. Para decidir qual centroide esta mais perto (uso em
    // cluster.c), comparar o quadrado da distancia da o mesmo resultado
    // que comparar a distancia real (sqrt nao muda a ordem) --> e evita um
    // sqrt() extra por par (ponto, centroide) a cada iteracao, que e o
    // loop mais pesado do algoritmo. Quem precisar da distancia "real"
    // (nenhum lugar precisa hoje) deve aplicar sqrt() no resultado.

    double soma = 0.0;

    for (int i = 0; i < numFeatures; i++) {

        double diferenca = ponto1[i] - ponto2[i];
        soma += diferenca * diferenca;
    }

    return soma;
}