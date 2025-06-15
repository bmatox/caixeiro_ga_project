```markdown
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

### 1. Pré-requisitos
Certifique-se de ter o Python 3 e as seguintes bibliotecas instaladas. Um arquivo `requirements.txt` poderia conter:
```
numpy
matplotlib
```
Para instalar, execute:
```bash
pip install -r requirements.txt
```
ou
```bash
pip install numpy matplotlib
```

### 2. Estrutura de Pastas
O projeto espera a seguinte estrutura de pastas para funcionar corretamente:
```
/seu_projeto/
|
├── data/
│   └── CaixeiroGruposGA.csv
|
└── src/
    └── main.py
```

### 3. Uso
Execute o script a partir da pasta raiz do projeto (`/seu_projeto/`).

**Execução Principal (Com Amostragem de Pontos):**
Para selecionar 40 pontos por região, conforme o requisito do trabalho:
```bash
python src/main.py --n_per_region 40
```

**Execução com Todos os Pontos:**
Para usar o dataset completo (161 pontos), simplesmente omita o argumento `--n_per_region`:
```bash
python src/main.py
```

**Execução Customizada:**
Para testar outros parâmetros do GA, passe-os como argumentos.
```bash
python src/main.py --pop_size 300 --max_gens 10000 --mutation_prob 0.05
```

## ⚙️ Parâmetros Configuráveis

Todos os parâmetros do algoritmo podem ser ajustados via linha de comando. Os valores padrão representam a melhor configuração encontrada durante os testes.

| Argumento | Descrição | Padrão no Código |
| :--- | :--- | :--- |
| `--data_path` | Caminho para o arquivo CSV dos pontos. | `data/CaixeiroGruposGA.csv` |
| `--origin_idx` | Índice da linha que representa a origem no CSV. | `0` |
| `--n_per_region`| Nº de pontos a selecionar por região (30 a 60). | `None` (Usa todos) |
| `--pop_size` | Tamanho da população (N). | `250` |
| `--max_gens` | Número máximo de gerações. | `8000` |
| `--tournament_k` | Número de indivíduos no torneio de seleção. | `3` |
| `--crossover_prob` | Probabilidade de aplicar o crossover. | `0.92` |
| `--mutation_prob`| Probabilidade de mutação por indivíduo (swap). | `0.01` |
| `--elitism` | Ativar (`True`) ou desativar (`False`) o elitismo. | `True` |
| `--elite_size` | Quantidade de indivíduos de elite a preservar. | `1` |
| `--no_improve_window`| Janela de gerações sem melhoria para parada. | `350` |

## 📊 Análise de Resultados

Foram realizados múltiplos experimentos para analisar o comportamento do algoritmo e a importância de seus parâmetros.

* **Diagnóstico Inicial:** Configurações iniciais (e.g., população de 100, mutação de 1%) levaram a uma **convergência prematura**, com o algoritmo ficando preso em ótimos locais de alto custo (~4901).
* **Ajuste de Parâmetros:** Aumentar a taxa de mutação (e.g., para 5%) mostrou-se crucial para aumentar a **exploração** e evitar a estagnação. Subsequentemente, foi observado que com uma população maior e mais "paciência" (`no_improve_window`), uma taxa de mutação menor (`1%`) permitia um **refinamento** mais preciso da solução.
* **Conclusão dos Testes:** A melhor configuração encontrada durante os testes (`pop_size=250`, `mutation_prob=0.01`, `no_improve_window=350`, etc.) alcançou um custo de **~2188**. Embora seja uma melhora drástica em relação aos resultados iniciais, este valor não superou o custo da heurística **Nearest Neighbor (~1813)**, ilustrando o trade-off entre a complexidade de um algoritmo e a qualidade da solução para um determinado orçamento computacional.

## 🖼️ Exemplo de Saída

* **Curva de Convergência:** O gráfico abaixo mostra a melhora do custo da melhor solução ao longo das gerações.
    *(Insira aqui o seu gráfico `convergence.png` ou visualize ao executar)*

* **Rota Final em 3D:** A visualização abaixo mostra a rota final encontrada pelo algoritmo.
    *(Insira aqui o seu gráfico `route_3d.png` ou visualize ao executar)*

---
##### Autores
* Bruno Matos e João Pedro Rego
* **Data:** 14/06/2025
* **Versão:** 1.0

```
