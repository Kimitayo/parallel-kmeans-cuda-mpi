#!/bin/bash
#SBATCH --job-name=teste_k
#SBATCH --output=teste_k_%j.out
#SBATCH --error=teste_k_%j.err
#SBATCH --partition=gpu-8-v100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

cd "$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "Rodando no host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

module purge
module load softwares/tau/2.35.1
module load cuda/12.6.2-gcc-8.5.0-ehvl44c

echo ""
echo "Dataset atual (confirme que e' o D2, 106497 linhas):"
wc -l vinhos.csv

echo ""
echo "========== Compilando sequencial ==========" 
cd sequencial
gcc -fopenmp -O2 kmeans.c ingest.c normalize.c distance.c centroid.c cluster.c convergence.c sse.c -o kmeans_seq -lm
cd ..

echo ""
echo "========== Compilando CUDA (otimizada) =========="
cd cuda
nvcc -O2 -rdc=true -arch=sm_70 cluster.cu distance.cu centroid.cu convergence.cu cuda_utils.cu flatten.cu ingest.cu normalize.cu sse.cu kmeans.cu -o kmeans_cuda -lm
cd ..

echo ""
echo "========== Testando k=2, k=5, k=8 (Sequencial) =========="
cd sequencial
for k in 2 5 8; do
    echo ""
    echo "--- Sequencial k=$k ---"
    ./kmeans_seq $k 100
done
cd ..

echo ""
echo "========== Testando k=2, k=5, k=8 (CUDA otimizada) =========="
cd cuda
for k in 2 5 8; do
    echo ""
    echo "--- CUDA k=$k ---"
    ./kmeans_cuda $k 100
done
cd ..

echo ""
echo "Fim: $(date)"
echo "=========================================="
