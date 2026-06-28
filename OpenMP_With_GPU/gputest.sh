#!/bin/bash
#SBATCH --partition=gpu-8-v100 
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:12:00
#SBATCH --job-name=gputest
#SBATCH --output=gputest-%j.out

ulimit -s unlimited

# Carrega o compilador da NVIDIA
module load compilers/nvidia/nvhpc/24.11

nvc -mp=gpu -gpu=managed -O3 kmeans.c ingest.c normalize.c distance.c centroid.c cluster.c convergence.c sse.c -o kmeans -lm

# Executa o arquivo gerado (sem o .exe)
./kmeans
