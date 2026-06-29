#!/bin/bash
#SBATCH --job-name=kmeans_benchmark
#SBATCH --output=kmeans_benchmark_%j.out
#SBATCH --error=kmeans_benchmark_%j.err
#SBATCH --partition=gpu-8-v100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

cd "$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "Rodando no host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Diretorio do job: $SLURM_SUBMIT_DIR"
echo "Data de inicio: $(date)"
echo "=========================================="

module purge

module load softwares/tau/2.35.1
module load cuda/12.6.2-gcc-8.5.0-ehvl44c

echo ""
echo "Modulos carregados:"
module list

echo ""
echo "Versao do Python:"
python3 --version

echo ""
echo "Versao do GCC:"
gcc --version | head -n 1

echo ""
echo "Versao do NVCC:"
nvcc --version

echo ""
echo "GPU disponivel:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

echo ""
echo "Arquivos na raiz do projeto:"
ls

echo ""
echo "Iniciando benchmark..."
echo "=========================================="

python3 benchmark.py

echo "=========================================="
echo "Benchmark finalizado."
echo "Data de fim: $(date)"
echo "=========================================="

echo ""
echo "Arquivos gerados:"
ls -lh

echo ""
echo "Conteudo inicial de resultados.csv, se existir:"
if [ -f resultados.csv ]; then
    head -n 20 resultados.csv
else
    echo "resultados.csv nao foi gerado."
fi