#ifndef TIPOS
#define TIPOS

typedef struct {
    double *features;
    int qualidade; // Guardado apenas para análise posterior
    int id; // Guardado apenas para análise posterior
    int cluster; // Indica a qual grupo este vinho pertence atualmente
} Vinho;

typedef struct {
    Vinho *dados;
    int linhas; 
    int colunas;
} Dataset;

typedef struct {
    int linhas;
    int colunas;    
} Dimensoes;

typedef struct {
    double *features; // O centro do grupo (média geométrica)
} Centroide;

#endif