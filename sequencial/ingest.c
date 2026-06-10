#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ingest.h" 

Dimensoes contarDimensoesCSV(const char *nomeArquivo) {
    FILE *arquivo = fopen(nomeArquivo, "r");
    if (!arquivo) {
        printf("Erro ao abrir o arquivo %s para contagem.\n", nomeArquivo);
        exit(1);
    }

    char linha[1024];
    Dimensoes dimensoes = {0, 0};

    // Pula o cabeçalho
    if (!fgets(linha, sizeof(linha), arquivo)) {
        fclose(arquivo);
        return dimensoes; // Retorna dimensões zeradas se o arquivo estiver vazio
    }

    int colunas = 1;
    for (int i = 0; linha[i] != '\0'; i++) {
        if (linha[i] == ',') {
            colunas++;
        }
    }

    dimensoes.colunas = colunas - 2; // Subtrai 2 porque as colunas 'quality' e 'Id' não são features de cálculo
    
    // Conta o restante das linhas com dados
    while (fgets(linha, sizeof(linha), arquivo)) {
        // Ignora linhas puramente vazias se houver
        if (strlen(linha) > 1) {
            dimensoes.linhas++;
        }
    }



    fclose(arquivo);
    return dimensoes;
}


// ATENÇÃO: ASSUME QUE AS DUAS ÚLTIMAS COLUNAS SÃO 'quality' E 'Id', E QUE O RESTANTE SÃO FEATURES
void lerCSV(const char *nomeArquivo, Vinho dataset[], int numColunas, int numLinhas) {
    FILE *arquivo = fopen(nomeArquivo, "r");
    if (!arquivo) {
        printf("Erro ao abrir o arquivo %s para leitura.\n", nomeArquivo);
        exit(1);
    }

    char linha[2048]; // Buffer para armazenar cada linha do CSV
    // Pula o cabeçalho
    char* cab = fgets(linha, sizeof(linha), arquivo);

    for (int cont = 0; cont < numLinhas; ) {
        // Se o arquivo acabar antes do esperado (proteção contra arquivos mal formatados)
        if (!fgets(linha, sizeof(linha), arquivo)) {
            break; 
        }
        // Remove quebra de linha
        linha[strcspn(linha, "\n")] = 0;

        // Ignora linhas vazias (sem incrementar o 'cont')
        if (strlen(linha) <= 1) continue; 

        // Aloca dinamicamente o array de features para ESTE vinho específico
        dataset[cont].features = (double *)malloc(numColunas * sizeof(double));
        if (!dataset[cont].features) {
            printf("Erro ao alocar features para a linha %d.\n", cont);
            exit(1);
        }

        // Extrai as N colunas de características (features)
        char *dado = strtok(linha, ",");
        for (int i = 0; i < numColunas; i++) {
            if (dado != NULL) {
                dataset[cont].features[i] = atof(dado); // Converte string para double
                dado = strtok(NULL, ",");
            }
        }

        // Como as duas últimas colunas são qualidade e ID, pega os próximos dados
        if (dado != NULL) {
            dataset[cont].qualidade = atoi(dado); // Converte para int
            dado = strtok(NULL, ",");
        }
        if (dado != NULL) {
            dataset[cont].id = atoi(dado);
        }

        // Todos os vinhos começam sem cluster definido
        dataset[cont].cluster = -1; 
        cont++;
    }
    
    fclose(arquivo);
}

Dataset* carregarDados(const char *nomeArquivo) {
    Dimensoes dimensoes = contarDimensoesCSV(nomeArquivo);
    int totalLinhas = dimensoes.linhas;
    int totalColunas = dimensoes.colunas;

    if (totalLinhas == 0 || totalColunas <= 0) {
        printf("O arquivo está vazio ou inválido.\n");
        exit(1);
    }

    Dataset *dataset = (Dataset*)malloc(sizeof(Dataset));
    Vinho *dados = (Vinho*)malloc(totalLinhas * sizeof(Vinho));
    
    if (!dados || !dataset) {
        printf("Falha na alocação do Dataset.\n");
        exit(1);
    }
    lerCSV(nomeArquivo, dados, totalColunas, totalLinhas);

    dataset->dados = dados;
    dataset->linhas = totalLinhas;
    dataset->colunas = totalColunas;

    return dataset;
}