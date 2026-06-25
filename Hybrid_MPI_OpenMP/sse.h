#ifndef SSE_H
#define SSE_H

#include <mpi.h>
#include "tipos.h"

double calcularSSE(Dataset *dataset, Centroide *centroides, MPI_Comm comm);

#endif
