# K-Means Paralelo: Análise de Desempenho em CPU e GPU

Projeto da disciplina **IMD1116 – Computação de Alto Desempenho** (UFRN), focado em implementar e analisar o algoritmo de agrupamento K-Means em diferentes paradigmas de programação paralela, com benchmarks reais executados no supercomputador **NPAD/UFRN** (GPU Tesla V100, 32 núcleos de CPU).

---

## 👥 Equipe

| Nome | 
|---|
| Clara Virgínia Nascimento Santos | 
| Cleiane Clementino Bondade | 
| Felipe Augusto Lemos Barreto | 
| Vinícius Barbosa Ventura Mergulhão |

---

## 📌 Visão Geral

O projeto implementa o K-Means em **4 versões**, todas em linguagem C, aplicadas ao **Wine Quality Dataset** (Kaggle). O objetivo é comparar o desempenho entre abordagens sequencial, paralela em CPU e paralela em GPU, avaliando escalabilidade, speedup e eficiência.

### Versões implementadas

| # | Versão | Pasta | Tecnologia |
|---|---|---|---|
| 1 | Sequencial (baseline) | `sequencial/` | C puro |
| 2 | Paralela CPU | `Hybrid_MPI_OpenMP/` | MPI + OpenMP |
| 3 | Paralela GPU (offloading) | `OpenMP_With_GPU/` | OpenMP target (`#pragma omp target`) |
| 4 | Paralela GPU (nativa) | `cuda/` | CUDA (cuDa Malloc Managed) |

---

## 📊 Dataset

- **Wine Quality Dataset** — [Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- **Linguagem:** C
- **Dataset médio (D2):** 106.497 linhas, 11 features
- **Dataset grande (D3):** 250.000 linhas, 11 features (gerado a partir do D2 com variação aleatória controlada)
- **k = 3 clusters** (padrão nos benchmarks de desempenho)
- **Semente fixa:** `srand(42)` em todas as versões — garante comparação justa

---

## 🖥️ Ambiente de Execução

| Componente | Especificação |
|---|---|
| Supercomputador | NPAD/UFRN (`sc2.npad.ufrn.br`) |
| GPU | Tesla V100-SXM2-16GB (Compute Capability 7.0) |
| Núcleos CPU | 32 (por nó alocado) |
| Partição SLURM | `gpu-8-v100` |
| CUDA | 12.6.2 |
| NVIDIA HPC SDK | 24.11 (para OpenMP-GPU) |
| MPI | OpenMPI |

---

## 📈 Resultados — Dataset Médio (D2, 106.497 linhas, k=3)

### Escalabilidade Forte — MPI+OpenMP (tamanho fixo, varia processos×threads)

| Processos (MPI) | Threads (OpenMP) | Recursos totais | Tempo (s) | Speedup | Eficiência |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | 1 | 0,436 | 1,00x | 100,0% |
| 2 | 1 | 2 | 0,300 | 1,45x | 72,7% |
| 1 | 2 | 2 | 0,401 | 1,09x | 54,4% |
| 4 | 1 | 4 | 0,274 | **1,59x** | 39,8% |
| 2 | 2 | 4 | 0,295 | 1,48x | 37,0% |
| 8 | 1 | 8 | 0,285 | 1,53x | 19,1% |
| 4 | 2 | 8 | 0,327 | 1,33x | 16,7% |
| 16 | 1 | 16 | 0,358 | 1,22x | 7,6% |
| 8 | 2 | 16 | 0,298 | **1,47x** | 9,2% |
| 16 | 2 | 32 | 0,251 | **1,74x** | 5,4% |
| 32 | 1 | 32 | 0,396 | 1,10x | 3,4% |
| 32 | 2 | 64 | 2,718 | 0,16x | 0,2% |

> ⚠️ O colapso em 32×2 (64 recursos) é esperado: excede os 32 núcleos reais do nó, gerando competição por CPU.

### Escalabilidade Fraca — todas as versões (tamanho cresce proporcionalmente aos recursos)

| Versão | 6.656 linhas | 13.312 | 26.624 | 53.248 | 106.497 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Sequencial** | 0,022s | 0,041s | 0,138s | 0,132s | 0,444s |
| **MPI+OpenMP** | 0,020s | 0,032s | 0,079s | 0,140s | 0,288s |
| **CUDA otimizada** | 0,024s | 0,267s | 0,729s | 0,106s | 0,393s |
| **CUDA modular** | 0,264s | 0,297s | 0,518s | 0,404s | 0,869s |
| **OpenMP-GPU** | 0,278s | 0,296s | 0,362s | 0,421s | 0,608s |

### Comparação GPU — CUDA otimizada vs CUDA modular vs OpenMP-GPU (D2)

| Tamanho | CUDA otimizada | CUDA modular | OpenMP-GPU |
|---|:---:|:---:|:---:|
| 6.656 | **0,024s** | 0,264s | 0,278s |
| 13.312 | **0,267s** | 0,297s | 0,296s |
| 26.624 | 0,729s | **0,518s** | 0,362s |
| 53.248 | **0,106s** | 0,404s | 0,421s |
| 106.497 | **0,393s** | 0,869s | 0,608s |

---

## 📈 Resultados — Dataset Grande (D3, 250.000 linhas, k=3)

### Escalabilidade Forte — MPI+OpenMP

| Processos × Threads | Recursos | Tempo (s) | Speedup |
|:---:|:---:|:---:|:---:|
| 1×1 | 1 | 0,713 | 1,00x |
| 2×1 | 2 | 0,649 | 1,10x |
| 4×1 | 4 | 0,666 | 1,07x |
| 2×2 | 4 | 0,642 | 1,11x |
| 8×1 | 8 | 0,791 | 0,90x |
| 16×1 | 16 | 0,781 | 0,91x |
| 16×2 | 32 | 0,772 | 0,92x |
| 32×1 | 32 | 1,140 | 0,63x |
| 32×2 | 64 | 1,882 | 0,38x |

> ⚠️ No D3 (só 19 iterações), o overhead MPI supera o ganho em quase todas as combinações — o paralelo não escala bem quando o algoritmo converge rápido.

### Escalabilidade Fraca — todas as versões (D3)

| Versão | 15.625 | 31.250 | 62.500 | 125.000 | 250.000 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Sequencial** | 0,061s | 0,094s | 0,190s | 0,474s | 0,717s |
| **MPI+OpenMP** | 0,058s | 0,084s | 0,170s | 0,514s | 0,667s |
| **CUDA otimizada** | 0,380s | 0,198s | **0,164s** | **0,339s** | **0,615s** |
| **CUDA modular** | 0,328s | 0,357s | 0,470s | 0,882s | 1,148s |
| **OpenMP-GPU** | 0,307s | 0,358s | 0,476s | 0,705s | 1,126s |

---

## 🔬 Efeito do número de clusters (k) — D2, CUDA otimizada

| k | Seq. tempo | Seq. iterações | CUDA tempo | CUDA iterações | Speedup GPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 0,265s | 18 | 0,250s | 17 | 0,94x |
| 3 | 0,440s | 61 | 0,393s | 32 | 1,12x |
| 5 | 0,703s | 70 | 0,416s | 38 | 1,69x |
| 8 | 0,940s | 65 | 0,249s | 50 | **3,78x** |

> 📌 A vantagem da GPU cresce significativamente com k: a cada iteração, cada ponto calcula distância para todos os k centroides — trabalho embaraçosamente paralelo que a GPU absorve muito bem.

---

## ✅ Validação Cruzada (SSE)

Todas as versões foram validadas comparando o **SSE final** (Sum of Squared Errors) entre si. Sequencial, MPI+OpenMP e OpenMP-GPU produzem resultados **numericamente idênticos**. A CUDA otimizada apresenta uma divergência mínima (na 3ª/4ª casa decimal) causada pelo não-determinismo inerente ao `atomicAdd` em ponto flutuante — o resultado final é matematicamente equivalente.

| Versão | SSE (D2, k=3) | Iterações |
|---|:---:|:---:|
| Sequencial | 9290,065373 | 61 |
| MPI+OpenMP | 9290,065373 | 61 |
| CUDA otimizada | 9290,077490 | 32 |
| CUDA modular | 9290,065373 | 61 |
| OpenMP-GPU | 9290,065373 | 61 |

---

## ⚙️ Compilação e Execução no NPAD

### Sequencial
```bash
cd sequencial
gcc -fopenmp -O2 kmeans.c ingest.c normalize.c distance.c centroid.c cluster.c convergence.c sse.c -o kmeans_seq -lm
./kmeans_seq 3 100
```

### MPI + OpenMP
```bash
cd Hybrid_MPI_OpenMP
mpicc -fopenmp -O2 kmeans.c ingest.c normalize.c distance.c centroid.c cluster.c convergence.c sse.c -o kmeans_hibrido -lm
mpirun -np 4 ./kmeans_hibrido 3 100   # ajuste -np conforme desejado
```

### CUDA (otimizada)
```bash
cd cuda
nvcc -O2 -rdc=true -arch=sm_70 cluster.cu distance.cu centroid.cu convergence.cu cuda_utils.cu flatten.cu ingest.cu normalize.cu sse.cu kmeans.cu -o kmeans_cuda -lm
./kmeans_cuda 3 100
```

### OpenMP-GPU
```bash
cd OpenMP_With_GPU
# requer NVIDIA HPC SDK: module load compilers/nvidia/nvhpc/24.11
nvc -mp=gpu -gpu=managed -O3 kmeans.c ingest.c normalize.c distance.c centroid.c cluster.c convergence.c sse.c -o kmeans_openmp_gpu -lm
./kmeans_openmp_gpu 3 100
```

### Benchmark completo (NPAD, via SLURM)
```bash
sbatch benchmark_completo.slurm
```

---

## 📁 Estrutura do Repositório

```
parallel-kmeans-cuda-mpi/
├── sequencial/              # Versão baseline (C puro)
├── Hybrid_MPI_OpenMP/       # Versão CPU paralela (MPI + OpenMP)
├── cuda/                    # Versão GPU nativa (CUDA otimizada)
├── OpenMP_With_GPU/         # Versão GPU por offloading (OpenMP target)
├── vinhos.csv               # Dataset principal (D2, 106.497 linhas)
├── benchmark.py             # Script de benchmark automatizado
├── benchmark_completo.slurm # Job SLURM para execução no NPAD
└── resultados.csv           # Resultados das medições
```
