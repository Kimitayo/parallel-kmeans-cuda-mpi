#include <math.h>

#include "convergence.h"

int convergiu(Centroide *antigos, Centroide *novos, int k, int numFeatures, double tolerancia) {
    double diferencaTotal = 0.0;

    for (int c = 0; c < k; c++) {
        for (int f = 0; f < numFeatures; f++) {
            double diferenca = antigos[c].features[f] - novos[c].features[f];

            diferencaTotal += fabs(diferenca);
        }
    }

    return diferencaTotal < tolerancia;
}