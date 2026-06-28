#!/bin/bash
#SBATCH --job-name=kmeans_benchmark
#SBATCH --output=benchmark_%j.log
#SBATCH --error=benchmark_%j.err
#SBATCH --partition=gpu-8-v100
#SBATCH --nodes=1                 # Como seu teste vai até 8 processos e 16 threads (oversubscription), 1 nó intel-128 geralmente basta.
#SBATCH --ntasks=8                # Máximo de processos MPI que você vai disparar (COMBINACOES_MPI_OPENMP vai até 8)
#SBATCH --cpus-per-task=2         # Para permitir que os processos MPI usem threads OpenMP sem brigar pelo mesmo núcleo
#SBATCH --time=0-0:10            # Limite de 1 hora de execução
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1

# Carregar os módulos necessários do supercomputador (Ajuste conforme os comandos do seu cluster)
# Exemplo comum:
# module load gcc
# module load openmpi
echo "=== Forçando Caminhos do CUDA (NPAD SDK) ==="

# Injeta a pasta bin do CUDA 12.6 no PATH para o Python encontrar o 'nvcc'
export PATH=/opt/npad/shared/compilers/nvidia_hpc_sdk/24.11/Linux_x86_64/24.11/cuda/12.6/bin:$PATH

# Injeta também a pasta de bibliotecas (lib64) essencial para o executável rodar depois
export LD_LIBRARY_PATH=/opt/npad/shared/compilers/nvidia_hpc_sdk/24.11/Linux_x86_64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

echo "Verificando se o nvcc responde..."
nvcc --version

# Mantém o resto do seu script abaixo
cd $SLURM_SUBMIT_DIR
python3 benchmark.py


# ======================================

# Garante que o Slurm mude para o diretório de onde você enviou o comando sbatch
cd $SLURM_SUBMIT_DIR
echo "=== Iniciando Benchmark Automático via Slurm ==="
echo "Nós alocados: $SLURM_JOB_NODELIST"

# Executa o seu script Python diretamente
python3 benchmark.py

echo "=== Benchmark Finalizado ==="