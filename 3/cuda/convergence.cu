#include <math.h>

#include <cuda_runtime.h>

#include "convergence.cuh"
#include "cuda_utils.cuh"

__global__
void kernelVerificarConvergencia(const double *centroides, const double *centroidesAntigos, int totalValores, double tolerancia, int *convergiu) {

    int indice = indiceGlobal();

    if (indice >= totalValores)
        return;

    if (fabs(centroides[indice] - centroidesAntigos[indice]) > tolerancia) {
        atomicExch(convergiu, 0);
    }
}

bool convergiu(const double *centroides, const double *centroidesAntigos, int *bufferConvergencia, int k, int numFeatures, double tolerancia) {

    int totalValores = k * numFeatures;

    *bufferConvergencia = 1;

    int blocos = calcularQuantidadeBlocos(totalValores, THREADS_POR_BLOCO);

    kernelVerificarConvergencia<<<blocos, THREADS_POR_BLOCO>>>(centroides, centroidesAntigos, totalValores, tolerancia, bufferConvergencia);

    CUDA_KERNEL_CHECK();

    return *bufferConvergencia == 1;
}