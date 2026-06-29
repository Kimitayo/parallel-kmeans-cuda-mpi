#!/usr/bin/env python3
"""
    raiz_do_projeto/
    ├── vinhos.csv                 
    ├── benchmark.py                <- este arquivo
    ├── sequencial/
    ├── Hybrid_MPI_OpenMP/
    ├── cuda/                       (só roda se existir + tiver nvcc)
    └── OpenMP_With_GPU/            (so roda se existir + tiver nvc, NVIDIA HPC SDK)

"""

import csv
import os
import random
import re
import shutil
import subprocess
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))


# CONFIGURACOES GERAIS 


K_CLUSTERS = 3
MAX_ITER = 100
REPETICOES = 5  # cada configuracao roda 5x; o resultado salvo e' a MEDIA

# Nome do compilador C. No NPAD normalmente "gcc" já funciona. 
COMPILADOR_C = "gcc"

# Arquitetura da GPU para o nvcc (CUDA). V100 = sm_70, T4 = sm_75, A100 = sm_80.
CUDA_ARCH = "sm_70"


# NUMERO DE NUCLEOS 
NUCLEOS_NO = 32


def gerar_combinacoes_strong_scaling(nucleos):
    """Gera combinacoes (processos, threads) cobrindo 1 ate `nucleos`
    recursos totais em potencias de 2, nos 3 perfis (so' processos, so'
    threads, e misto), mais 1 combinacao de oversubscription de proposito
    (2x o numero de nucleos) para mostrar o efeito de saturacao."""
    combinacoes = []
    potencia = 1
    while potencia <= nucleos:
        combinacoes.append((potencia, 1))         # so' MPI
        if potencia > 1:
            combinacoes.append((1, potencia))     # so' OpenMP
        meio = max(1, potencia // 2)
        if meio > 1 and potencia // meio > 1 and (meio, potencia // meio) not in combinacoes:
            combinacoes.append((meio, potencia // meio))  # misto
        potencia *= 2

    # 1 ponto de oversubscription de proposito (2x os nucleos reais)
    combinacoes.append((nucleos, 2))

    # remove duplicatas mantendo ordem
    vistos = set()
    unicas = []
    for c in combinacoes:
        if c not in vistos:
            vistos.add(c)
            unicas.append(c)
    return unicas


COMBINACOES_STRONG = gerar_combinacoes_strong_scaling(NUCLEOS_NO)

# Niveis de recurso para ESCALABILIDADE FRACA
NIVEIS_WEAK = [n for n in (1, 2, 4, 8, 16) if n <= NUCLEOS_NO]

TEMPO_REGEX = re.compile(r"Tempo de execucao:\s*([\d.,]+)\s*segundos")
SSE_REGEX = re.compile(r"Qualidade do agrupamento \(SSE\):\s*([\d.,]+)")
ITER_REGEX = re.compile(r"Convergiu na iteracao\s*(\d+)")



# Geracao de subconjuntos do dataset oficial (para weak scaling e curva de
# tamanho). Usa AMOSTRAGEM (nao duplicacao/jitter) do dataset real


def ler_dataset_oficial(caminho_csv):
    with open(caminho_csv, "r", encoding="utf-8") as f:
        leitor = csv.reader(f)
        cabecalho = next(leitor)
        linhas = list(leitor)
    return cabecalho, linhas


def escrever_subconjunto(caminho_csv, cabecalho, linhas, tamanho, seed=42):
    """Escreve em `caminho_csv` uma amostra determinística de `tamanho`
    linhas tiradas de `linhas` (sem repor). Se tamanho >= len(linhas),
    escreve todas (na ordem original)."""
    if tamanho >= len(linhas):
        amostra = linhas
    else:
        rng = random.Random(seed)
        amostra = rng.sample(linhas, tamanho)

    with open(caminho_csv, "w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f, lineterminator="\n")
        escritor.writerow(cabecalho)
        escritor.writerows(amostra)


def calcular_tamanhos_weak(n_oficial, niveis):
    """Tamanhos proporcionais aos niveis de recurso (ex: niveis 1,2,4,8,16
    -> fracoes 1/16,2/16,4/16,8/16,16/16 do dataset oficial)."""
    maior_nivel = max(niveis)
    tamanhos = {}
    for nivel in niveis:
        fracao = nivel / maior_nivel
        tamanhos[nivel] = max(500, int(n_oficial * fracao))
    return tamanhos




def extrair_tempo(saida_stdout, comando_str):
    m = TEMPO_REGEX.search(saida_stdout)
    if not m:
        print("[ERRO] Nao encontrei 'Tempo de execucao' na saida de: %s" % comando_str)
        print("----- saida completa -----")
        print(saida_stdout)
        print("---------------------------")
        sys.exit(1)
    return float(m.group(1).replace(",", "."))


def extrair_sse(saida_stdout):
    """Retorna o SSE como float, ou None se nao encontrar (nao trata como
    erro fatal -- SSE e' so' para validacao cruzada, nao essencial)."""
    m = SSE_REGEX.search(saida_stdout)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))


def extrair_iteracoes(saida_stdout):
    m = ITER_REGEX.search(saida_stdout)
    if not m:
        return None
    return int(m.group(1))


def rodar(comando, cwd, env=None, repeticoes=REPETICOES):
    """Roda o comando `repeticoes` vezes e retorna (tempo_medio, sse_medio,
    iteracoes_media). sse/iteracoes podem vir como None se o programa nao
    imprimir esses dados (nao e' erro fatal, so' fica de fora do CSV)."""
    tempos = []
    sses = []
    iteracoes_lista = []
    comando_str = " ".join(comando)

    for i in range(repeticoes):
        resultado = subprocess.run(
            comando, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True
        )

        if resultado.returncode != 0:
            print("[ERRO] Falha ao executar: %s" % comando_str)
            print(resultado.stderr)
            sys.exit(1)

        tempo = extrair_tempo(resultado.stdout, comando_str)
        sse = extrair_sse(resultado.stdout)
        iteracoes = extrair_iteracoes(resultado.stdout)

        tempos.append(tempo)
        if sse is not None:
            sses.append(sse)
        if iteracoes is not None:
            iteracoes_lista.append(iteracoes)

        info_extra = ""
        if sse is not None:
            info_extra += "  SSE=%.4f" % sse
        if iteracoes is not None:
            info_extra += "  iter=%d" % iteracoes
        print("    tentativa %d/%d: %.6fs%s" % (i + 1, repeticoes, tempo, info_extra))

    media_tempo = sum(tempos) / len(tempos)
    media_sse = (sum(sses) / len(sses)) if sses else None
    media_iter = (sum(iteracoes_lista) / len(iteracoes_lista)) if iteracoes_lista else None

    print("    -> media: %.6fs" % media_tempo)


    if len(set(round(s, 4) for s in sses)) > 1:
        print("    [AVISO] SSE variou entre as %d repeticoes: %s"
              % (len(sses), ["%.4f" % s for s in sses]))

    return media_tempo, media_sse, media_iter


def compilar(comando, cwd, nome_etapa):
    print("Compilando %s..." % nome_etapa)
    resultado = subprocess.run(
        comando, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True
    )
    if resultado.returncode != 0:
        print("[ERRO] Falha ao compilar %s:" % nome_etapa)
        print(resultado.stderr)
        sys.exit(1)
    print("  -> OK")


def tem_comando(nome):
    return shutil.which(nome) is not None


def registrar(linhas_csv, versao, escalabilidade, processos, threads, linhas_dataset,
              tempo, t_referencia, recursos, sse_medio=None, iteracoes_media=None):
    speedup = t_referencia / tempo if tempo > 0 else 0
    eficiencia = speedup / recursos if recursos > 0 else None
    linhas_csv.append({
        "versao": versao,
        "escalabilidade": escalabilidade,
        "processos_mpi": processos,
        "threads_openmp": threads,
        "linhas_dataset": linhas_dataset,
        "tempo_segundos": round(tempo, 6),
        "speedup": round(speedup, 4),
        "eficiencia": round(eficiencia, 4) if eficiencia is not None else "",
        "sse_medio": round(sse_medio, 6) if sse_medio is not None else "",
        "iteracoes_medias": round(iteracoes_media, 2) if iteracoes_media is not None else "",
    })
    return speedup, eficiencia



# MAIN


def main():
    caminho_vinhos = os.path.join(RAIZ, "vinhos.csv")
    if not os.path.isfile(caminho_vinhos):
        print("[ERRO] vinhos.csv nao encontrado na raiz do projeto.")
        sys.exit(1)

    with open(caminho_vinhos, "r", encoding="utf-8") as f:
        conteudo_original_vinhos = f.read()

    cabecalho, linhas_oficiais = ler_dataset_oficial(caminho_vinhos)
    n_oficial = len(linhas_oficiais)
    print("Dataset oficial: %d linhas (sera preservado; o script restaura no final)\n" % n_oficial)

    tamanhos_weak = calcular_tamanhos_weak(n_oficial, NIVEIS_WEAK)
    print("Niveis de recurso para escalabilidade fraca -> tamanho de dataset:")
    for nivel, tamanho in tamanhos_weak.items():
        print("  %d recurso(s) -> %d linhas" % (nivel, tamanho))
    print()

    linhas_csv = []

    try:

        # 1) SEQUENCIAL -- baseline no tamanho OFICIAL (escalabilidade forte)

        pasta_seq = os.path.join(RAIZ, "sequencial")
        if not os.path.isdir(pasta_seq):
            print("[ERRO] Pasta 'sequencial/' nao encontrada.")
            sys.exit(1)

        compilar(
            [COMPILADOR_C, "-fopenmp", "-O2", "kmeans.c", "ingest.c", "normalize.c",
             "distance.c", "centroid.c", "cluster.c", "convergence.c", "sse.c",
             "-o", "kmeans_seq", "-lm"],
            cwd=pasta_seq, nome_etapa="SEQUENCIAL"
        )

        print("\n=== SEQUENCIAL | tamanho oficial (%d linhas) ===" % n_oficial)
        t_seq_oficial, sse_seq_oficial, iter_seq_oficial = rodar(
            ["./kmeans_seq", str(K_CLUSTERS), str(MAX_ITER)], cwd=pasta_seq)

        registrar(linhas_csv, "Sequencial", "forte", 1, 1, n_oficial,
                  t_seq_oficial, t_seq_oficial, 1, sse_seq_oficial, iter_seq_oficial)


        # 2) MPI + OPENMP -- ESCALABILIDADE FORTE (tamanho fixo = oficial,
        #    varia processos x threads)

        pasta_hibrida = os.path.join(RAIZ, "Hybrid_MPI_OpenMP")
        tem_mpi = os.path.isdir(pasta_hibrida) and tem_comando("mpicc") and tem_comando("mpirun")

        if not os.path.isdir(pasta_hibrida):
            print("\n[AVISO] Pasta 'Hybrid_MPI_OpenMP/' nao encontrada -- pulando.")
        elif not tem_comando("mpicc") or not tem_comando("mpirun"):
            print("\n[AVISO] mpicc/mpirun nao encontrados no PATH -- pulando MPI+OpenMP.")
        else:
            compilar(
                ["mpicc", "-fopenmp", "-O2", "kmeans.c", "ingest.c", "normalize.c",
                 "distance.c", "centroid.c", "cluster.c", "convergence.c", "sse.c",
                 "-o", "kmeans_hibrido", "-lm"],
                cwd=pasta_hibrida, nome_etapa="MPI + OpenMP"
            )

            print("\n--- Escalabilidade FORTE (tamanho fixo = %d linhas) ---" % n_oficial)
            for processos, threads in COMBINACOES_STRONG:
                recursos = processos * threads
                print("\n=== MPI+OpenMP [FORTE] | processos=%d threads=%d (tamanho=%d) ==="
                      % (processos, threads, n_oficial))

                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(threads)

                comando = ["mpirun", "--oversubscribe", "-np", str(processos),
                           "./kmeans_hibrido", str(K_CLUSTERS), str(MAX_ITER)]

                tempo, sse, iteracoes = rodar(comando, cwd=pasta_hibrida, env=env)
                speedup, eficiencia = registrar(
                    linhas_csv, "MPI+OpenMP", "forte", processos, threads,
                    n_oficial, tempo, t_seq_oficial, recursos, sse, iteracoes
                )
                print("  speedup=%.3fx  eficiencia=%.1f%%" % (speedup, eficiencia * 100))


        # 3) ESCALABILIDADE FRACA -- para TODAS as versoes disponiveis, no
        #    mesmo conjunto de tamanhos (proporcionais aos niveis de recurso)

        print("\n\n########## ESCALABILIDADE FRACA (tamanho varia com os recursos) ##########")

        pasta_cuda = os.path.join(RAIZ, "cuda")
        tem_cuda = os.path.isdir(pasta_cuda) and tem_comando("nvcc")
        if os.path.isdir(pasta_cuda) and tem_comando("nvcc"):
            compilar(
                ["nvcc", "-O2", "-rdc=true", "-arch=" + CUDA_ARCH,
                 "cluster.cu", "distance.cu", "centroid.cu", "convergence.cu",
                 "cuda_utils.cu", "flatten.cu", "ingest.cu", "normalize.cu",
                 "sse.cu", "kmeans.cu",
                 "-o", "kmeans_cuda", "-lm"],
                cwd=pasta_cuda, nome_etapa="CUDA"
            )
        elif os.path.isdir(pasta_cuda):
            print("[AVISO] Pasta 'cuda/' existe mas nvcc nao foi encontrado -- pulando CUDA.")

        pasta_gpu_omp = os.path.join(RAIZ, "OpenMP_With_GPU")
        tem_gpu_omp = os.path.isdir(pasta_gpu_omp) and tem_comando("nvc")
        if os.path.isdir(pasta_gpu_omp) and tem_comando("nvc"):
            compilar(
                ["nvc", "-mp=gpu", "-gpu=managed", "-O3",
                 "kmeans.c", "ingest.c", "normalize.c", "distance.c",
                 "centroid.c", "cluster.c", "convergence.c", "sse.c",
                 "-o", "kmeans_openmp_gpu", "-lm"],
                cwd=pasta_gpu_omp, nome_etapa="OpenMP-GPU"
            )
        elif os.path.isdir(pasta_gpu_omp):
            print("[AVISO] Pasta 'OpenMP_With_GPU/' existe mas 'nvc' (NVIDIA HPC SDK) "
                  "nao foi encontrado -- pulando OpenMP-GPU. Confirme 'module load' no job SLURM.")

        for nivel in sorted(NIVEIS_WEAK):
            tamanho = tamanhos_weak[nivel]
            print("\n----- Tamanho desta rodada: %d linhas (nivel de recurso=%d) -----" % (tamanho, nivel))

            # gera o subconjunto e escreve no vinhos.csv da raiz (compartilhado
            # por todas as pastas de versao, via "../vinhos.csv")
            escrever_subconjunto(caminho_vinhos, cabecalho, linhas_oficiais, tamanho)

            # ---- Sequencial neste tamanho (referencia local para o speedup) ----
            print("\n=== Sequencial [FRACA] | tamanho=%d ===" % tamanho)
            t_seq_local, sse_seq_local, iter_seq_local = rodar(
                ["./kmeans_seq", str(K_CLUSTERS), str(MAX_ITER)], cwd=pasta_seq)
            registrar(linhas_csv, "Sequencial", "fraca", 1, 1, tamanho,
                      t_seq_local, t_seq_local, 1, sse_seq_local, iter_seq_local)

            # ---- MPI+OpenMP: processos = nivel de recurso, 1 thread cada ----
            if tem_mpi:
                print("\n=== MPI+OpenMP [FRACA] | processos=%d threads=1 | tamanho=%d ===" % (nivel, tamanho))
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = "1"
                comando = ["mpirun", "--oversubscribe", "-np", str(nivel),
                           "./kmeans_hibrido", str(K_CLUSTERS), str(MAX_ITER)]
                tempo, sse, iteracoes = rodar(comando, cwd=pasta_hibrida, env=env)
                speedup, eficiencia = registrar(
                    linhas_csv, "MPI+OpenMP", "fraca", nivel, 1, tamanho,
                    tempo, t_seq_local, nivel, sse, iteracoes
                )
                print("  speedup=%.3fx  eficiencia=%.1f%% (ideal p/ weak scaling: eficiencia ~100%%)"
                      % (speedup, eficiencia * 100))

            # ---- CUDA (sem "recursos" variaveis -- so' a curva por tamanho) ----
            if tem_cuda:
                print("\n=== CUDA | tamanho=%d ===" % tamanho)
                tempo, sse, iteracoes = rodar(["./kmeans_cuda", str(K_CLUSTERS), str(MAX_ITER)], cwd=pasta_cuda)
                registrar(linhas_csv, "CUDA", "fraca", "", "", tamanho,
                          tempo, t_seq_local, 1, sse, iteracoes)

            # ---- OpenMP-GPU (idem, curva por tamanho p/ comparar com CUDA) ----
            if tem_gpu_omp:
                print("\n=== OpenMP-GPU | tamanho=%d ===" % tamanho)
                tempo, sse, iteracoes = rodar(["./kmeans_openmp_gpu", str(K_CLUSTERS), str(MAX_ITER)], cwd=pasta_gpu_omp)
                registrar(linhas_csv, "OpenMP-GPU", "fraca", "", "", tamanho,
                          tempo, t_seq_local, 1, sse, iteracoes)

    finally:
        # SEMPRE restaura o dataset oficial original, mesmo se algo falhar acima
        with open(caminho_vinhos, "w", encoding="utf-8") as f:
            f.write(conteudo_original_vinhos)
        print("\n(vinhos.csv restaurado para o conteudo oficial original)")

    # -----------------------------------------------------------------------
    # Salva o CSV final
    # -----------------------------------------------------------------------
    caminho_csv = os.path.join(RAIZ, "resultados.csv")
    campos = ["versao", "escalabilidade", "processos_mpi", "threads_openmp",
              "linhas_dataset", "tempo_segundos", "speedup", "eficiencia",
              "sse_medio", "iteracoes_medias"]

    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(linhas_csv)

    print("\n✅ Resultados salvos em: %s" % caminho_csv)
    print("   (%d linhas no total)" % len(linhas_csv))


if __name__ == "__main__":
    main()
