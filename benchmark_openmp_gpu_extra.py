#!/usr/bin/env python3
"""
benchmark_openmp_gpu_extra.py

Roda o binario kmeans_openmp_gpu (ja compilado, na pasta OpenMP_With_GPU/)
nos MESMOS 5 tamanhos de dataset que o benchmark.py ja usou pra CUDA e
MPI+OpenMP na escalabilidade fraca, e ANEXA essas linhas ao resultados.csv
existente -- nao recria o arquivo do zero.

Por que isso existe separado do benchmark.py:
O modulo nvhpc (necessario pra COMPILAR o OpenMP-GPU) entra em conflito com
o mpicc do MPI+OpenMP quando os dois estao carregados juntos. Por isso, no
job SLURM, a compilacao do OpenMP-GPU continua acontecendo numa etapa
separada (com nvhpc carregado, igual ja fazem hoje). Mas EXECUTAR o binario
ja compilado nao precisa do nvhpc carregado -- so' precisa dele pra
compilar. Este script so' executa (nao compila), e por isso pode rodar
tanto com nvhpc carregado quanto sem.

Pre-requisitos antes de chamar este script:
  - resultados.csv ja deve existir (gerado pelo benchmark.py na Parte 1),
    com as linhas "Sequencial,fraca,..." para os 5 tamanhos (usadas aqui
    como referencia pra calcular o speedup do OpenMP-GPU)
  - o binario ./OpenMP_With_GPU/kmeans_openmp_gpu ja deve estar compilado
  - vinhos.csv na raiz deve estar com o conteudo OFICIAL (este script
    restaura esse conteudo no final, mesmo se der erro no meio)

Uso (chamado da RAIZ do projeto, depois de compilar o OpenMP-GPU):
    python3 benchmark_openmp_gpu_extra.py
"""

import csv
import os
import random
import re
import subprocess
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))

K_CLUSTERS = 3
MAX_ITER = 100
REPETICOES = 5

# Mesmos niveis de recurso usados no benchmark.py para a escalabilidade
# fraca -- PRECISA bater com o NIVEIS_WEAK de la' (NUCLEOS_NO ali define
# o limite). Se mudou um, mude o outro tambem.
NIVEIS_WEAK = [1, 2, 4, 8, 16]

PASTA_GPU = os.path.join(RAIZ, "OpenMP_With_GPU")
CAMINHO_VINHOS = os.path.join(RAIZ, "vinhos.csv")
CAMINHO_RESULTADOS = os.path.join(RAIZ, "resultados.csv")

TEMPO_REGEX = re.compile(r"Tempo de execucao:\s*([\d.,]+)\s*segundos")
SSE_REGEX = re.compile(r"Qualidade do agrupamento \(SSE\):\s*([\d.,]+)")
ITER_REGEX = re.compile(r"Convergiu na iteracao\s*(\d+)")


def ler_dataset_oficial(caminho_csv):
    with open(caminho_csv, "r", encoding="utf-8") as f:
        leitor = csv.reader(f)
        cabecalho = next(leitor)
        linhas = list(leitor)
    return cabecalho, linhas


def escrever_subconjunto(caminho_csv, cabecalho, linhas, tamanho, seed=42):
    """IDENTICO ao escrever_subconjunto() do benchmark.py -- mesma semente,
    mesmo metodo de amostragem -- para garantir que o OpenMP-GPU seja
    testado exatamente nos MESMOS dados que CUDA/MPI+OpenMP foram."""
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
    """IDENTICO ao calcular_tamanhos_weak() do benchmark.py."""
    maior_nivel = max(niveis)
    tamanhos = {}
    for nivel in niveis:
        fracao = nivel / maior_nivel
        tamanhos[nivel] = max(500, int(n_oficial * fracao))
    return tamanhos


def ler_tempos_sequencial_por_tamanho(caminho_resultados):
    """Le o resultados.csv ja existente e devolve {linhas_dataset: tempo}
    para as linhas 'Sequencial,fraca' (usado como referencia de speedup)."""
    referencias = {}
    if not os.path.isfile(caminho_resultados):
        return referencias

    with open(caminho_resultados, "r", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        for linha in leitor:
            if linha.get("versao") == "Sequencial" and linha.get("escalabilidade") == "fraca":
                try:
                    tamanho = int(linha["linhas_dataset"])
                    tempo = float(linha["tempo_segundos"])
                    referencias[tamanho] = tempo
                except (KeyError, ValueError):
                    continue
    return referencias


def extrair_tempo(saida_stdout):
    m = TEMPO_REGEX.search(saida_stdout)
    return float(m.group(1).replace(",", ".")) if m else None


def extrair_sse(saida_stdout):
    m = SSE_REGEX.search(saida_stdout)
    return float(m.group(1).replace(",", ".")) if m else None


def extrair_iteracoes(saida_stdout):
    m = ITER_REGEX.search(saida_stdout)
    return int(m.group(1)) if m else None


def rodar_openmp_gpu(repeticoes=REPETICOES):
    """Roda o binario ja compilado `repeticoes` vezes; retorna as medias
    de (tempo, sse, iteracoes)."""
    tempos, sses, iteracoes_lista = [], [], []
    binario = os.path.join(PASTA_GPU, "kmeans_openmp_gpu")

    if not os.path.isfile(binario):
        print("[ERRO] Binario nao encontrado: %s" % binario)
        print("       Compile o OpenMP-GPU antes de chamar este script.")
        sys.exit(1)

    for i in range(repeticoes):
        resultado = subprocess.run(
            ["./kmeans_openmp_gpu", str(K_CLUSTERS), str(MAX_ITER)],
            cwd=PASTA_GPU,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True
        )

        if resultado.returncode != 0:
            print("[ERRO] Falha ao executar kmeans_openmp_gpu:")
            print(resultado.stderr)
            sys.exit(1)

        tempo = extrair_tempo(resultado.stdout)
        sse = extrair_sse(resultado.stdout)
        iteracoes = extrair_iteracoes(resultado.stdout)

        if tempo is None:
            print("[ERRO] Nao encontrei 'Tempo de execucao' na saida.")
            print(resultado.stdout)
            sys.exit(1)

        tempos.append(tempo)
        if sse is not None:
            sses.append(sse)
        if iteracoes is not None:
            iteracoes_lista.append(iteracoes)

        extra = ""
        if sse is not None:
            extra += "  SSE=%.4f" % sse
        if iteracoes is not None:
            extra += "  iter=%d" % iteracoes
        print("    tentativa %d/%d: %.6fs%s" % (i + 1, repeticoes, tempo, extra))

    media_tempo = sum(tempos) / len(tempos)
    media_sse = (sum(sses) / len(sses)) if sses else None
    media_iter = (sum(iteracoes_lista) / len(iteracoes_lista)) if iteracoes_lista else None

    if len(set(round(s, 4) for s in sses)) > 1:
        print("    [AVISO] SSE variou entre as %d repeticoes: %s"
              % (len(sses), ["%.4f" % s for s in sses]))

    return media_tempo, media_sse, media_iter


def main():
    if not os.path.isfile(CAMINHO_VINHOS):
        print("[ERRO] vinhos.csv nao encontrado em %s" % RAIZ)
        sys.exit(1)

    with open(CAMINHO_VINHOS, "r", encoding="utf-8") as f:
        conteudo_original_vinhos = f.read()

    cabecalho, linhas_oficiais = ler_dataset_oficial(CAMINHO_VINHOS)
    n_oficial = len(linhas_oficiais)

    tamanhos_weak = calcular_tamanhos_weak(n_oficial, NIVEIS_WEAK)
    tempos_seq_referencia = ler_tempos_sequencial_por_tamanho(CAMINHO_RESULTADOS)

    print("Dataset oficial: %d linhas" % n_oficial)
    print("Tamanhos a testar: %s\n" % sorted(tamanhos_weak.values()))

    linhas_novas = []

    try:
        for nivel in sorted(NIVEIS_WEAK):
            tamanho = tamanhos_weak[nivel]
            print("\n=== OpenMP-GPU | tamanho=%d (nivel=%d) ===" % (tamanho, nivel))

            escrever_subconjunto(CAMINHO_VINHOS, cabecalho, linhas_oficiais, tamanho)

            tempo, sse, iteracoes = rodar_openmp_gpu()

            t_seq = tempos_seq_referencia.get(tamanho)
            if t_seq is not None:
                speedup = t_seq / tempo
                eficiencia = speedup  # GPU: 1 "recurso" (nao ha processos/threads pra dividir)
                print("  speedup=%.3fx (vs sequencial nesse tamanho)" % speedup)
            else:
                print("  [AVISO] Nao achei o tempo sequencial de referencia para "
                      "tamanho=%d no resultados.csv -- speedup/eficiencia ficam vazios."
                      % tamanho)
                speedup = None
                eficiencia = None

            linhas_novas.append({
                "versao": "OpenMP-GPU",
                "escalabilidade": "fraca",
                "processos_mpi": "",
                "threads_openmp": "",
                "linhas_dataset": tamanho,
                "tempo_segundos": round(tempo, 6),
                "speedup": round(speedup, 4) if speedup is not None else "",
                "eficiencia": round(eficiencia, 4) if eficiencia is not None else "",
                "sse_medio": round(sse, 6) if sse is not None else "",
                "iteracoes_medias": round(iteracoes, 2) if iteracoes is not None else "",
            })

    finally:
        with open(CAMINHO_VINHOS, "w", encoding="utf-8") as f:
            f.write(conteudo_original_vinhos)
        print("\n(vinhos.csv restaurado para o conteudo oficial original)")

    # ---- Anexa (nao sobrescreve) ao resultados.csv existente ----
    campos = ["versao", "escalabilidade", "processos_mpi", "threads_openmp",
              "linhas_dataset", "tempo_segundos", "speedup", "eficiencia",
              "sse_medio", "iteracoes_medias"]

    arquivo_novo = not os.path.isfile(CAMINHO_RESULTADOS)
    with open(CAMINHO_RESULTADOS, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        if arquivo_novo:
            writer.writeheader()
        writer.writerows(linhas_novas)

    print("\n✅ %d linhas do OpenMP-GPU adicionadas a: %s" % (len(linhas_novas), CAMINHO_RESULTADOS))


if __name__ == "__main__":
    main()
