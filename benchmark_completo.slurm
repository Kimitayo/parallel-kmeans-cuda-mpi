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

# ---------------------------------------------------------------------------
# Carrega TODOS os modulos de uma vez so' (CUDA + MPI + NVIDIA HPC SDK),
# antes de chamar o benchmark.py. Antes isso era feito em 2 partes
# separadas (CUDA primeiro, nvhpc depois), o que fazia o benchmark.py NAO
# encontrar o 'nvc' e pular o OpenMP-GPU inteiro do sweep automatico --
# por isso ele so' tinha 1 resultado manual, sem SSE/iteracoes nem curva
# de tamanhos. Carregando tudo de uma vez, o benchmark.py detecta o 'nvc'
# e roda o OpenMP-GPU nos mesmos 5 tamanhos que o CUDA, com SSE/iteracoes
# incluidos automaticamente no resultados.csv.
# ---------------------------------------------------------------------------
module purge
module load softwares/tau/2.35.1
module load cuda/12.6.2-gcc-8.5.0-ehvl44c
module load compilers/nvidia/nvhpc/24.11

echo ""
echo "Modulos carregados:"
module list

echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

echo ""
echo "Verificando nvcc (CUDA) e nvc (OpenMP-GPU):"
which nvcc
which nvc

echo ""
echo "========== Rodando benchmark.py (Sequencial + MPI/OpenMP + CUDA + OpenMP-GPU) =========="
python3 benchmark.py

echo ""
echo "========== RESULTADOS FINAIS =========="
cat resultados.csv

echo ""
echo "Fim: $(date)"
echo "=========================================="
