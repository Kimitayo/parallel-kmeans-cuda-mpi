#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <mpi.h>
#include "tipos.h"

// Normaliza todas as features do dataset
void normalizarDataset(Dataset *dataset, MPI_Comm comm);

#endif
