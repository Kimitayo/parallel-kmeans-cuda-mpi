#!/bin/bash
#SBATCH --job-name=kmeans_completo
#SBATCH --output=kmeans_completo_%j.out
#SBATCH --error=kmeans_completo_%j.err
#SBATCH --partition=gpu-8-v100
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

cd "$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "Rodando no host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Diretorio: $SLURM_SUBMIT_DIR"
echo "Inicio: $(date)"
echo "=========================================="

echo ""
echo "========== PARTE 1: Sequencial + MPI/OpenMP + CUDA =========="

module purge
module load softwares/tau/2.35.1
module load cuda/12.6.2-gcc-8.5.0-ehvl44c

echo ""
echo "Modulos da parte 1:"
module list

echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

echo ""
echo "Rodando benchmark.py (Sequencial + MPI/OpenMP + CUDA)..."
python3 benchmark.py

echo ""
echo "========== PARTE 2: OpenMP-GPU com nvc =========="

module purge
module load softwares/tau/2.35.1
module load cuda/12.6.2-gcc-8.5.0-ehvl44c
module load compilers/nvidia/nvhpc/24.11

echo ""
echo "Modulos da parte 2:"
module list

echo ""
echo "Verificando nvc:"
which nvc
nvc --version

echo ""
echo "Entrando em OpenMP_With_GPU..."
cd "$SLURM_SUBMIT_DIR/OpenMP_With_GPU"

echo ""
echo "Compilando OpenMP-GPU..."
nvc -mp=gpu -gpu=managed -O3 \
    kmeans.c ingest.c normalize.c distance.c centroid.c cluster.c convergence.c sse.c \
    -o kmeans_openmp_gpu -lm

echo ""
echo "Voltando para a raiz do projeto..."
cd "$SLURM_SUBMIT_DIR"

echo ""
echo "Rodando OpenMP-GPU nos mesmos 5 tamanhos do CUDA/MPI+OpenMP..."
echo "(nao precisa do nvhpc carregado para EXECUTAR, so' precisou dele para compilar)"
python3 benchmark_openmp_gpu_extra.py

echo ""
echo "========== RESULTADOS FINAIS =========="
cat resultados.csv

echo ""
echo "Fim: $(date)"
echo "=========================================="
