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
echo "Iniciando as 5 execucoes do K-Means na GPU..."
echo "------------------------------------------------"

> metricas_execucao.txt

for i in {1..4}; do
    echo -n "Execucao $i: "
    
    SAIDA=$(./kmeans)
    
    TEMPO=$(echo "$SAIDA" | grep "Tempo de execucao:" | awk '{print $4}')
    ITER=$(echo "$SAIDA" | grep "Convergiu na iteracao" | awk '{print $4}')
    SSE=$(echo "$SAIDA" | grep "Qualidade do agrupamento (SSE):" | awk '{print $5}')
    SSEM=$(echo "$SAIDA" | grep "SSE medio:" | awk '{print $3}')
    
    echo "$TEMPO seg | $ITER iteracoes | SSE: $SSE"
    
    echo "$TEMPO $ITER $SSE $SSEM" >> metricas_execucao.txt
done

echo "------------------------------------------------"
echo "Resultados Finais (Media de 5 execucoes):"
echo "------------------------------------------------"

awk '{
    soma_tempo += $1;
    soma_iter  += $2;
    soma_sse   += $3;
    soma_ssem  += $4;
} END { 
    if (NR > 0) {
        printf "Tempo de execucao    : %.6f segundos\n", soma_tempo / NR;
        printf "Iteracoes (Converg.) : %.1f iteracoes\n", soma_iter / NR;
        printf "Qualidade (SSE)      : %.6f\n", soma_sse / NR;
        printf "SSE Medio            : %.6f\n", soma_ssem / NR;
    }
}' metricas_execucao.txt

echo "------------------------------------------------"