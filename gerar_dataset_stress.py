#!/usr/bin/env python3
"""
gerar_dataset_stress.py - Gera uma versao "stress test" do vinhos.csv,
duplicando as linhas originais ate' atingir o tamanho desejado.

Objetivo: ter um dataset grande o suficiente para o trabalho de cada
iteracao do K-means superar o overhead de criar threads/processos -->
util para VALIDAR que a paralelizacao funciona, antes de rodar no NPAD.


Uso:
    python3 gerar_dataset_stress.py [linhas_desejadas]

    (padrao: 100000 linhas, se nao especificar)
"""

import csv
import random
import sys

ARQUIVO_ORIGINAL = "vinhos.csv"
ARQUIVO_SAIDA = "vinhos_stress.csv"
LINHAS_DESEJADAS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000

# pequeno ruido aleatorio (+-0.5%) em cada feature numerica, so' pra evitar
# que o dataset fique com milhares de pontos EXATAMENTE identicos (o que
# deixaria o clustering artificialmente "facil demais" e poderia mascarar
# o comportamento real do algoritmo)
RUIDO_RELATIVO = 0.005

random.seed(42)  # reprodutibilidade


def main():
    with open(ARQUIVO_ORIGINAL, "r", encoding="utf-8") as f:
        leitor = csv.reader(f)
        cabecalho = next(leitor)
        linhas_originais = list(leitor)

    n_original = len(linhas_originais)
    print(f"Dataset original: {n_original} linhas")
    print(f"Gerando: {LINHAS_DESEJADAS} linhas (~{LINHAS_DESEJADAS / n_original:.1f}x)")

    # indices das colunas que sao features numericas (todas, exceto
    # 'quality' e 'Id', que sao as duas ultimas colunas)
    indices_features = list(range(len(cabecalho) - 2))

    with open(ARQUIVO_SAIDA, "w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f, lineterminator="\n")
        escritor.writerow(cabecalho)

        novo_id = 0
        linhas_escritas = 0

        while linhas_escritas < LINHAS_DESEJADAS:
            for linha_original in linhas_originais:
                if linhas_escritas >= LINHAS_DESEJADAS:
                    break

                nova_linha = list(linha_original)

                # aplica ruido pequeno em cada feature numerica
                for idx in indices_features:
                    valor = float(nova_linha[idx])
                    fator = 1.0 + random.uniform(-RUIDO_RELATIVO, RUIDO_RELATIVO)
                    nova_linha[idx] = f"{valor * fator:.6f}"

                # Id sempre unico e sequencial (nao e' usado pelo algoritmo,
                # mas mantemos por organizacao)
                nova_linha[-1] = str(novo_id)

                escritor.writerow(nova_linha)
                novo_id += 1
                linhas_escritas += 1

    print(f"\n✅ Arquivo gerado: {ARQUIVO_SAIDA} ({linhas_escritas} linhas)")
    print("\nPara usar: troque temporariamente o nome do arquivo lido em")
    print('cada kmeans.c/kmeans.cu (a linha "../vinhos.csv") para')
    print('"../vinhos_stress.csv", ou crie uma copia do vinhos.csv original')
    print("renomeada e troque de volta depois do teste.")


if __name__ == "__main__":
    main()
