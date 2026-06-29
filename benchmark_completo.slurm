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
echo "Rodando benchmark.py sem nvhpc..."
python3 benchmark.py

echo ""
echo "========== PARTE 2: OpenMP GPU com nvc =========="

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
echo "Rodando OpenMP-GPU no dataset oficial..."
./kmeans_openmp_gpu 3 100 | tee "$SLURM_SUBMIT_DIR/openmp_gpu_saida.txt"

echo ""
echo "Extraindo tempo/SSE/iteracoes OpenMP-GPU..."

TEMPO_OMP_GPU=$(grep "Tempo de execucao:" "$SLURM_SUBMIT_DIR/openmp_gpu_saida.txt" | tail -n 1 | sed -E 's/.*Tempo de execucao:[[:space:]]*([0-9.,]+).*/\1/' | tr ',' '.')
SSE_OMP_GPU=$(grep "Qualidade do agrupamento (SSE):" "$SLURM_SUBMIT_DIR/openmp_gpu_saida.txt" | tail -n 1 | sed -E 's/.*SSE\):[[:space:]]*([0-9.,]+).*/\1/' | tr ',' '.')
ITER_OMP_GPU=$(grep "Convergiu na iteracao" "$SLURM_SUBMIT_DIR/openmp_gpu_saida.txt" | tail -n 1 | sed -E 's/.*iteracao[[:space:]]*([0-9]+).*/\1/')

echo "Tempo OpenMP-GPU extraido: $TEMPO_OMP_GPU"
echo "SSE OpenMP-GPU extraido: $SSE_OMP_GPU"
echo "Iteracoes OpenMP-GPU extraidas: $ITER_OMP_GPU"

cd "$SLURM_SUBMIT_DIR"

echo ""
echo "Adicionando OpenMP-GPU ao resultados.csv..."

if [ -f resultados.csv ] && [ -n "$TEMPO_OMP_GPU" ]; then
    TEMPO_SEQ=$(awk -F',' '$1=="Sequencial" && $2=="forte" {print $6; exit}' resultados.csv)

    SPEEDUP=$(python3 - <<EOF
seq_time = float("$TEMPO_SEQ")
gpu_time = float("$TEMPO_OMP_GPU")
print(round(seq_time / gpu_time, 4))
EOF
)

    EFICIENCIA="$SPEEDUP"

    # CSV novo tem 10 colunas:
    # versao,escalabilidade,processos_mpi,threads_openmp,linhas_dataset,
    # tempo_segundos,speedup,eficiencia,sse_medio,iteracoes_medias
    echo "OpenMP-GPU,fraca,,,106497,$TEMPO_OMP_GPU,$SPEEDUP,$EFICIENCIA,$SSE_OMP_GPU,$ITER_OMP_GPU" >> resultados.csv

    echo "Linha adicionada:"
    tail -n 1 resultados.csv
else
    echo "Nao foi possivel adicionar OpenMP-GPU ao resultados.csv"
fi

echo ""
echo "========== RESULTADOS FINAIS =========="
cat resultados.csv

echo ""
echo "Fim: $(date)"
echo "=========================================="
