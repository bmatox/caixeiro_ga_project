# Algoritmo Genético para o Problema do Caixeiro Viajante 3D

Este projeto apresenta uma implementação em Python de um Algoritmo Genético (GA) para resolver uma instância do Problema do Caixeiro Viajante (TSP) em um espaço tridimensional. O objetivo é encontrar uma rota de custo subótimo para um drone que precisa visitar uma série de pontos e retornar à sua origem.

O script é flexível, permitindo tanto a utilização do conjunto de dados completo quanto a seleção de um subconjunto de pontos por região, conforme especificado nos requisitos do trabalho.

## ✨ Funcionalidades

* **Implementação de GA "do zero"**: Toda a lógica do algoritmo genético foi implementada utilizando apenas bibliotecas padrão do Python e NumPy.
* **Amostragem Configurável**: Permite selecionar um número específico de pontos por região a partir do arquivo de dados.
* **Representação por Permutação**: O cromossomo de cada indivíduo representa uma rota como uma permutação dos índices dos pontos a serem visitados.
* **Seleção por Torneio**: Implementa um operador de seleção com tamanho de torneio configurável para controlar a pressão seletiva.
* **Crossover Ordenado (OX1)**: Utiliza um operador de recombinação apropriado para problemas de permutação, garantindo que os filhos gerados sejam sempre rotas válidas.
* **Mutação de Troca (Swap)**: Aplica uma mutação que troca a posição de dois genes (cidades) aleatórios no cromossomo.
* **Elitismo**: Garante que os melhores indivíduos de uma geração sejam preservados na próxima.
* **Múltiplos Critérios de Parada**: O algoritmo pode parar ao atingir um número máximo de gerações, ao estagnar por falta de melhora, ou ao encontrar uma solução considerada "aceitável".
* **Visualização de Resultados**: Gera automaticamente um gráfico da curva de convergência (custo vs. geração) e um plot 3D da melhor rota encontrada.

## 🚀 Como Executar

### Pré-requisitos

Certifique-se de ter o Python 3 e as seguintes bibliotecas instaladas:

```bash
pip install numpy matplotlib
```

### Estrutura de Pastas

O arquivo de dados deve estar localizado em uma pasta `data/` na raiz do projeto:

```
/caixeiro_ga_project
│
├── data/
│   └── CaixeiroGruposGA.csv
│
└── src/          ← ou, se preferir, coloque `main.py` na raiz e ajuste comandos abaixo
    └── main.py
```

* Em muitos casos, o script único `main.py` fica em `src/main.py`.
* Se você preferir, pode ter apenas `main.py` na raiz (ajuste o comando de execução conforme explicado adiante).

### Execução

1. **Execução Principal (seleção de pontos por região)**
   Para selecionar um número específico de pontos por região (ex: 40), use o argumento `--n_per_region`.

   ```bash
   python src/main.py --n_per_region 40
   ```

   ou, se `main.py` estiver na raiz:

   ```bash
   python main.py --n_per_region 40
   ```

2. **Execução com Todos os Pontos**
   Para usar todos os pontos do arquivo CSV, simplesmente omita o argumento `--n_per_region`.

   ```bash
   python src/main.py
   ```

   ou

   ```bash
   python main.py
   ```

3. **Execução com Parâmetros do GA Customizados**
   Você pode customizar qualquer parâmetro do GA. Exemplo: 40 pontos por região, população de 300 e taxa de mutação de 10%:

   ```bash
   python src/main.py --n_per_region 40 --pop_size 300 --mutation_prob 0.1
   ```

   ou

   ```bash
   python main.py --n_per_region 40 --pop_size 300 --mutation_prob 0.1
   ```

## ⚙️ Parâmetros Configuráveis

| Argumento                | Descrição                                                    | Padrão no Código            |
| ------------------------ | ------------------------------------------------------------ | --------------------------- |
| `--data_path`            | Caminho para o arquivo CSV dos pontos.                       | `data/CaixeiroGruposGA.csv` |
| `--origin_idx`           | Índice da linha que representa a origem no CSV.              | `0`                         |
| `--n_per_region`         | Nº de pontos a selecionar por região (30 a 60).              | `None` (Usa todos)          |
| `--pop_size`             | Tamanho da população (N).                                    | `200`                       |
| `--max_gens`             | Número máximo de gerações.                                   | `8000`                      |
| `--tournament_k`         | Número de indivíduos no torneio de seleção.                  | `3`                         |
| `--crossover_prob`       | Probabilidade de aplicar o crossover.                        | `0.9`                       |
| `--mutation_prob`        | Probabilidade de mutação por indivíduo (swap).               | `0.01`                      |
| `--elitism`              | Ativar (True) ou desativar (False) o elitismo.               | `True`                      |
| `--elite_size`           | Quantidade de indivíduos de elite a preservar.               | `1`                         |
| `--no_improve_window`    | Janela de gerações sem melhoria para parada.                 | `250`                       |
| `--use_accept_criterion` | Usar critério de solução aceitável via heurística NN?        | `True`                      |
| `--tolerance`            | Tolerância relativa para solução aceitável (ex.: 0.05 = 5%). | `0.05`                      |
| `--save_plots`           | Salvar plots em disco? (True/False).                         | `True`                      |
| `--output_dir`           | Pasta para salvar plots (se `--save_plots=True`).            | `outputs`                   |
| `--verbose`              | Mostrar logs detalhados do GA (True/False).                  | `True`                      |
| `--n_runs`               | (Opcional) Nº de execuções independentes para estatísticas.  | `1`                         |

> **Observação**: Se o `main.py` estiver na raiz, substitua `python src/main.py` por `python main.py` nos exemplos acima.

## 📊 Análise de Resultados

Foram realizados múltiplos experimentos para analisar o comportamento do algoritmo e a importância de seus parâmetros:

* **Diagnóstico Inicial**

  * Configurações iniciais (população pequena, mutação de 1%) levaram a convergência prematura, com o algoritmo preso em ótimos locais de alto custo (\~4901).

* **Ajuste de Parâmetros**

  * Aumentar a taxa de mutação de 1% para 5% permitiu maior exploração, evitando ótimos locais e melhorando drasticamente o resultado final.
  * Aumentar a "paciência" do algoritmo (parâmetro `--no_improve_window`) para valores maiores (e.g., 250) deu tempo para a diversidade produzida pela mutação gerar soluções melhores.

* **Conclusão de Testes**

  * A melhor configuração encontrada durante os testes foi:

    * `pop_size=200`
    * `max_gens=5000`
    * `mutation_prob=0.05`
    * `no_improve_window=250`
  * Com essa combinação, obteve-se custo \~2477 na instância testada, embora a heurística Nearest Neighbor tivesse custo \~1813, ilustrando o trade-off entre complexidade de algoritmo e qualidade da solução para o orçamento computacional disponível.

### 🖼️ Exemplo de Saída

* **Curva de Convergência**
  O gráfico abaixo mostra a melhora do custo da melhor solução ao longo das gerações.
  *(Insira aqui o seu gráfico `convergence.png` ou visualize ao executar)*

* **Rota Final em 3D**
  A visualização abaixo mostra a rota final encontrada pelo algoritmo, conectando todos os pontos a partir da origem.
  *(Insira aqui o seu gráfico `route_3d.png` ou visualize ao executar)*

## 📁 Organização Sugerida de Código

Embora todo o código esteja em `main.py`, a lógica interna segue estas etapas:

1. **Carregamento e Amostragem de Pontos**

   * Lê o CSV, detecta header, extrai colunas X, Y, Z e grupo.
   * Se `--n_per_region` for especificado, amostra exatamente esse número de pontos de cada grupo (30 ≤ n ≤ 60), garantindo inclusão do ponto de origem.

2. **Preparação**

   * Calcula a matriz de distâncias Euclidianas 3D.
   * Determina índice da origem no conjunto final.
   * Executa heurística Nearest Neighbor para referência de custo.

3. **Algoritmo Genético Básico**

   * **Inicialização**: População de permutações aleatórias (excluindo origem).
   * **Seleção**: Torneio de tamanho configurável.
   * **Crossover**: Order Crossover (variação de dois pontos) para permutações sem repetição.
   * **Mutação**: Swap de dois genes com probabilidade configurável (padrão 1%).
   * **Elitismo**: Preserva os `elite_size` melhores indivíduos.
   * **Critérios de Parada**:

     * Máximo de gerações (`--max_gens`).
     * Sem melhora em `--no_improve_window` gerações.
     * (Opcional) Solução aceitável: custo ≤ heurística NN \* (1 - tol).

4. **Resultados e Visualizações**

   * Exibe logs de progresso (se `--verbose=True`).
   * Após término, plota curva de convergência e rota final em 3D e salva em `--output_dir`.

5. **Análise Estatística (Opcional)**

   * Com `--n_runs > 1`, executa o GA várias vezes, coleta gerações em que atinge solução aceitável e calcula moda/min/max para análise de estabilidade e impacto de parâmetros.

## 📋 Requisitos Atendidos

1. **Definição de pontos por região**: argumento `--n_per_region` com validação 30 ≤ n ≤ 60.
2. **Definição de N de indivíduos e gerações**: argumentos `--pop_size` e `--max_gens`.
3. **Operador de Seleção (Torneio)**: implementado em `tournament_selection` com `--tournament_k`.
4. **Recombinação de dois pontos sem repetição**: Order Crossover (OX) adequado para permutações.
5. **Mutação de swap 1%**: parâmetro `--mutation_prob=0.01` por padrão, troca de genes.
6. **Critérios de Parada**:

   * Máx gerações, sem melhora em janela (`--no_improve_window`), solução aceitável via heurística + tolerância (`--use_accept_criterion`, `--tolerance`).
   * A regra de “sem modificações genotípicas médias” não foi implementada explicitamente, pois a parada por estagnação na aptidão costuma ser suficiente em versão básica.
7. **Análise de Moda de Gerações e Elitismo**: oferecido via `--n_runs` e coleta de gerações atingidas, comparando com/sem elitismo (configurável via `--elitism` e `--elite_size`).

## 📦 Dependências

* Python 3.x
* NumPy
* Matplotlib

Um arquivo `requirements.txt` pode listar:

```
numpy
matplotlib
```

## 📖 Referências

* Conceitos de Algoritmos Genéticos: codificação em permutação, seleção por torneio, crossover ordenado, mutação swap, elitismo, critérios de parada.
* Heurística Nearest Neighbor para TSP como referência de solução aceitável.

---

##### Autor / Versão

* Desenvolvido por: Bruno Matos e João Pedro Rego
* Data: 14/06/2025
* Versão: 1.0 (versão básica conforme requisitos de sala)
