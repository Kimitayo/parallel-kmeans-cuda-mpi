#!/bin/bash
#SBATCH --job-name=kmeans_cuda
#SBATCH --partition=gpu              # CONFIRME com `sinfo` -- "gpu" e' citado no FAQ do NPAD,
                                      # mas pode haver mais de uma particao com GPU (ex: gpu-intel, gpu-amd)
#SBATCH --gres=gpu:1                 # solicita 1 GPU no no'
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00              # ajuste o tempo maximo esperado do job (formato dias-horas:minutos)
#SBATCH --output=kmeans_cuda_%j.out
#SBATCH --error=kmeans_cuda_%j.err

# ---------------------------------------------------------------------------
# ANTES de submeter este script (sbatch job_cuda.slurm), confirme manualmente
# no terminal do NPAD (ssh -p4422 cvnsantos@sc2.npad.ufrn.br):
#
#   sinfo                       -> ver o nome real da(s) particao(oes) com GPU
#   module avail cuda           -> ver a versao exata do modulo CUDA disponivel
#
# E troque os valores de --partition e module load abaixo de acordo.
# ---------------------------------------------------------------------------

echo "Rodando no host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv 2>&1

module load cuda   # CONFIRME o nome exato (ex: cuda/11.8, cuda/12.2 ...) com `module avail cuda`

cd "$SLURM_SUBMIT_DIR"

# Compilacao (poderia ser feita uma vez soa antes, deixei aqui para garantir
# que o binario sempre bate com a versao do CUDA carregada no no' alocado).
# Ajuste -arch=sm_XX conforme a GPU do no' (veja a saida do nvidia-smi acima
# e a tabela de compute capability no comentario no topo do kmeans.cu).
nvcc -O3 -arch=sm_70 kmeans.cu ingest.c normalize.cu distance.c centroid.cu cluster.cu convergence.c sse.c -o kmeans_cuda -lm

if [ $? -ne 0 ]; then
    echo "Falha na compilacao. Abortando."
    exit 1
fi

echo "=== Execucao K-means CUDA ==="
# Argumentos: [k_clusters] [max_iteracoes] -- ajuste conforme os testes do grupo
./kmeans_cuda 3 100

# ---------------------------------------------------------------------------
# Se for rodar a versao OpenMP-GPU (Felipe/Vinicius) no MESMO no' alocado,
# adicione as linhas de compilacao + execucao dela aqui embaixo, por exemplo:
#
#   echo "=== Execucao K-means OpenMP-GPU ==="
#   nvcc -O3 -arch=sm_70 -Xcompiler -fopenmp ../OpenMP_GPU/kmeans_openmp_gpu.c -o kmeans_openmp_gpu -lm
#   ./kmeans_openmp_gpu 3 100
#
# Assim os dois jobs compartilham a mesma alocacao de GPU/no', sem precisar
# entrar duas vezes na fila do SLURM.
# ---------------------------------------------------------------------------
