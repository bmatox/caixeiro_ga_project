# Projeto Caixeiro Viajante com Algoritmo Genético

## Estrutura do projeto

```
caixeiro_ga_project/
├── data/
│   └── CaixeiroGrupos.csv    # Coloque aqui o arquivo com os pontos
├── src/
│   ├── config.py             # Parâmetros do GA
│   ├── utils.py              # Funções auxiliares (carregamento de dados, heurística NN, cálculos)
│   ├── ga.py                 # Implementação do GA (versão básica conforme conceitos estudados)
│   ├── plotting.py           # Funções para plotar convergência e rota 3D
│   └── main.py               # Script principal para rodar o GA
└── README.md
```

## Instruções de Uso

1. Coloque o arquivo `CaixeiroGrupos.csv` na pasta `data/`.
2. Ajuste parâmetros em `src/config.py` conforme desejado (tamanho de população, gerações, torneio, etc.).
3. Execute:
   ```
   cd caixeiro_ga_project
   python3 -m src.main --origin_idx 0
   ```
   Substitua `--origin_idx` pelo índice do ponto de origem desejado.
4. O script irá:
   - Carregar os pontos.
   - Calcular heurística Nearest Neighbor para referência.
   - Executar o GA básico (codificação permutação, seleção torneio, crossover OX (dois pontos), mutação swap 1%, elitismo opcional, critérios de parada).
   - Exibir plots de convergência e rota final em 3D (salvos em arquivos PNG).
5. Consulte o código em `src/` para entender como cada parte foi implementada, alinhada com os algoritmos estudados em sala de aula.

---

## Mapeamento dos algoritmos usados

- **Codificação**: cada indivíduo é uma permutação dos índices de pontos (excluindo origem).
- **Função de aptidão**: custo da rota (soma de distâncias Euclidianas 3D entre pontos consecutivos, incluindo ida e volta à origem).
- **Inicialização**: população de permutações aleatórias.
- **Seleção**: torneio de tamanho k (definido em config).
- **Crossover**: variação de recombinação de dois pontos, implementada via Order Crossover (OX) para permutações sem repetição.
- **Mutação**: swap de dois genes com probabilidade pm (1% por indivíduo, configurável).
- **Elitismo**: cópia de Ne melhores indivíduos para a próxima geração.
- **Critérios de parada**:
  - Número máximo de gerações.
  - Sem melhora no melhor custo em `window` gerações.
  - (Opcional) solução aceitável baseada em heurística NN e tolerância epsilon.

Esses conceitos correspondem aos algoritmos fornecidos inicialmente: codificação, avaliação de aptidão, seleção por torneio, recombinação dois pontos (via máscara de OX), mutação uniforme (swap), e parâmetros de parada conforme regras estudadas.

## Observações

- Esta é a versão básica (não híbrida). A versão híbrida usa busca local adicional, mas foge do escopo trivial.
- Personalize `config.py` para experimentar diferentes valores.
- Para análise estatística de múltiplas execuções, adapte `main.py` para loop de execuções e colete dados conforme desejado.
