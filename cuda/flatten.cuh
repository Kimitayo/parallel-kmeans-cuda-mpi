#ifndef FLATTEN_H
#define FLATTEN_H

#include "tipos.h"

void flattenDataset(Dataset *dataset, double **features, int **clusters);

void liberarFlatten(double *features, int *clusters);

#endif