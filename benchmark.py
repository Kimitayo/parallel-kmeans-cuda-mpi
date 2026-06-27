#!/usr/bin/env python3
"""
benchmark.py - Roda as versoes do K-means, mede o tempo de execucao e
calcula speedup/eficiencia em relacao a versao sequencial.

Estrutura de pastas esperada (mesma do repositorio no GitHub):

    raiz_do_projeto/
    ├── vinhos.csv
    ├── sequencial/
    ├── Hybrid_MPI_OpenMP/
    ├── CUDA/                  (opcional -- so' roda se existir e tiver nvcc)
    └── benchmark.py           <- este arquivo, na raiz

Uso:
    python3 benchmark.py

Saida:
    resultados.csv  -- uma linha por execucao, com tempo/speedup/eficiencia
"""

import csv
import os
import re
import subprocess
import sys
import shutil

RAIZ = os.path.dirname(os.path.abspath(__file__))
K_CLUSTERS = 3
MAX_ITER = 100
REPETICOES = 3  # quantas vezes roda cada configuracao (usamos o melhor tempo, reduz ruido do SO)

# Nome do compilador C a usar. No Linux, "gcc" normal já funciona.
# No Mac, o "gcc" do sistema é na verdade um disfarce do Apple Clang e NÃO
# suporta -fopenmp -- troque para o gcc de verdade instalado via Homebrew
# (ex: "gcc-14"). Veja com `ls /opt/homebrew/bin | grep gcc-`.
COMPILADOR_C = "gcc-16"

# Combinações de processos MPI x threads OpenMP que serao testadas.
# Ajuste essa lista de acordo com o numero de nucleos da sua maquina/no'.
# Combinações de processos MPI x threads OpenMP que serao testadas.
# Ajustado para uma maquina com 8 nucleos: cobre strong scaling de 1 a 8
# recursos totais (processos x threads), e inclui 1 combinacao alem do
# limite (8x2=16) de proposito, pra mostrar o efeito de "oversubscription"
# (competicao por nucleo) no relatorio.
COMBINACOES_MPI_OPENMP = [
    # (processos_mpi, threads_openmp)  -> total de recursos usados
    (1, 1),   # 1  - baseline "paralelo" minimo (so' pra comparar overhead vs sequencial puro)
    (1, 2),   # 2
    (1, 4),   # 4
    (1, 8),   # 8  - so' OpenMP, sem MPI
    (2, 1),   # 2
    (2, 2),   # 4
    (2, 4),   # 8
    (4, 1),   # 4
    (4, 2),   # 8
    (8, 1),   # 8  - so' MPI, sem OpenMP
    (8, 2),   # 16 - DE PROPOSITO alem dos 8 nucleos (oversubscription)
]

TEMPO_REGEX = re.compile(r"Tempo de execucao:\s*([\d.]+)\s*segundos")


def tem_comando(nome):
    return shutil.which(nome) is not None


def extrair_tempo(saida_stdout, comando_str):
    m = TEMPO_REGEX.search(saida_stdout)
    if not m:
        print(f"[ERRO] Nao encontrei 'Tempo de execucao' na saida de: {comando_str}")
        print("----- saida completa -----")
        print(saida_stdout)
        print("---------------------------")
        sys.exit(1)
    return float(m.group(1))


def rodar(comando, cwd, env=None, repeticoes=REPETICOES):
    """Roda o comando `repeticoes` vezes e retorna o MENOR tempo (menos ruido)."""
    tempos = []
    comando_str = " ".join(comando)

    for i in range(repeticoes):
        resultado = subprocess.run(
            comando, cwd=cwd, env=env,
            capture_output=True, text=True
        )

        if resultado.returncode != 0:
            print(f"[ERRO] Falha ao executar: {comando_str}")
            print(resultado.stderr)
            sys.exit(1)

        tempo = extrair_tempo(resultado.stdout, comando_str)
        tempos.append(tempo)
        print(f"    tentativa {i + 1}/{repeticoes}: {tempo:.6f}s")

    return min(tempos)


def compilar(comando, cwd, nome_etapa):
    print(f"Compilando {nome_etapa}...")
    resultado = subprocess.run(comando, cwd=cwd, capture_output=True, text=True)
    if resultado.returncode != 0:
        print(f"[ERRO] Falha ao compilar {nome_etapa}:")
        print(resultado.stderr)
        sys.exit(1)
    print(f"  -> OK")


def main():
    linhas_csv = []

    # ------------------------------------------------------------------
    # 1) SEQUENCIAL (baseline)
    # ------------------------------------------------------------------
    pasta_seq = os.path.join(RAIZ, "sequencial")
    if not os.path.isdir(pasta_seq):
        print("[ERRO] Pasta 'sequencial/' nao encontrada na raiz do projeto.")
        sys.exit(1)

    compilar(
        [COMPILADOR_C, "-fopenmp", "-O2", "kmeans.c", "ingest.c", "normalize.c",
         "distance.c", "centroid.c", "cluster.c", "convergence.c", "sse.c",
         "-o", "kmeans_seq", "-lm"],
        cwd=pasta_seq, nome_etapa="versao sequencial"
    )

    print("\n=== Rodando SEQUENCIAL (baseline) ===")
    t_sequencial = rodar(["./kmeans_seq", str(K_CLUSTERS), str(MAX_ITER)], cwd=pasta_seq)
    print(f"  Tempo baseline (sequencial): {t_sequencial:.6f}s\n")

    linhas_csv.append({
        "versao": "Sequencial",
        "processos_mpi": 1,
        "threads_openmp": 1,
        "bloco_cuda": "",
        "tempo_segundos": t_sequencial,
        "speedup": 1.0,
        "eficiencia": 1.0,
    })

    # ------------------------------------------------------------------
    # 2) MPI + OpenMP
    # ------------------------------------------------------------------
    pasta_hibrida = os.path.join(RAIZ, "Hybrid_MPI_OpenMP")
    if not os.path.isdir(pasta_hibrida):
        print("[AVISO] Pasta 'Hybrid_MPI_OpenMP/' nao encontrada -- pulando essa etapa.")
    elif not tem_comando("mpicc") or not tem_comando("mpirun"):
        print("[AVISO] mpicc/mpirun nao encontrados no PATH -- pulando MPI+OpenMP.")
        print("        (No Ubuntu/Debian: sudo apt install libopenmpi-dev openmpi-bin)")
    else:
        compilar(
            ["mpicc", "-fopenmp", "-O2", "kmeans.c", "ingest.c", "normalize.c",
             "distance.c", "centroid.c", "cluster.c", "convergence.c", "sse.c",
             "-o", "kmeans_hibrido", "-lm"],
            cwd=pasta_hibrida, nome_etapa="versao MPI + OpenMP"
        )

        for processos, threads in COMBINACOES_MPI_OPENMP:
            print(f"\n=== MPI+OpenMP | processos={processos} threads={threads} ===")

            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(threads)

            comando = [
                "mpirun", "--oversubscribe", "-np", str(processos),
                "./kmeans_hibrido", str(K_CLUSTERS), str(MAX_ITER)
            ]

            tempo = rodar(comando, cwd=pasta_hibrida, env=env)

            recursos = processos * threads
            speedup = t_sequencial / tempo
            eficiencia = speedup / recursos

            print(f"  tempo={tempo:.6f}s  speedup={speedup:.3f}x  eficiencia={eficiencia:.1%}")

            linhas_csv.append({
                "versao": "MPI+OpenMP",
                "processos_mpi": processos,
                "threads_openmp": threads,
                "bloco_cuda": "",
                "tempo_segundos": tempo,
                "speedup": speedup,
                "eficiencia": eficiencia,
            })

    # ------------------------------------------------------------------
    # 3) CUDA (so' roda se houver nvcc -- normalmente so' no NPAD)
    # ------------------------------------------------------------------
    pasta_cuda = os.path.join(RAIZ, "CUDA")
    if not os.path.isdir(pasta_cuda):
        print("\n[AVISO] Pasta 'CUDA/' nao encontrada -- pulando essa etapa.")
    elif not tem_comando("nvcc"):
        print("\n[AVISO] nvcc nao encontrado no PATH -- pulando CUDA "
              "(normal se voce estiver rodando isso fora do NPAD).")
    else:
        compilar(
            ["nvcc", "-O3", "-arch=sm_60", "kmeans.cu", "ingest.c", "normalize.cu",
             "distance.c", "centroid.cu", "cluster.cu", "convergence.c", "sse.c",
             "-o", "kmeans_cuda", "-lm"],
            cwd=pasta_cuda, nome_etapa="versao CUDA"
        )

        print(f"\n=== CUDA ===")
        comando = ["./kmeans_cuda", str(K_CLUSTERS), str(MAX_ITER)]
        tempo = rodar(comando, cwd=pasta_cuda)

        speedup = t_sequencial / tempo
        # Para GPU nao existe um "numero de processadores" equivalente
        # direto; eficiencia aqui fica registrada como N/A no CSV --
        # use o speedup absoluto e a comparacao com OpenMP-GPU no relatorio.
        print(f"  tempo={tempo:.6f}s  speedup={speedup:.3f}x")

        linhas_csv.append({
            "versao": "CUDA",
            "processos_mpi": "",
            "threads_openmp": "",
            "bloco_cuda": "",
            "tempo_segundos": tempo,
            "speedup": speedup,
            "eficiencia": "",
        })

    # ------------------------------------------------------------------
    # Salva o CSV final
    # ------------------------------------------------------------------
    caminho_csv = os.path.join(RAIZ, "resultados.csv")
    campos = ["versao", "processos_mpi", "threads_openmp", "bloco_cuda",
              "tempo_segundos", "speedup", "eficiencia"]

    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(linhas_csv)

    print(f"\n✅ Resultados salvos em: {caminho_csv}")
    print(f"   ({len(linhas_csv)} linhas, baseline sequencial = {t_sequencial:.6f}s)")


if __name__ == "__main__":
    main()
