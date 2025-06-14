#!/usr/bin/env python3
"""
main.py: GA básico para Problema do Caixeiro-Viajante 3D usando numpy e matplotlib.
Salve este arquivo e chame:
    python main.py --data_path data/CaixeiroGrupos.csv --origin_idx 0

Argumentos configuráveis para parâmetros do GA, tolerância, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessário para 3D
import random
import argparse
import os
import sys

def load_csv_numpy(path):
    """
    Carrega CSV de pontos via numpy.genfromtxt, detectando header simples:
    Se a primeira linha gerar NaN, tenta skip_header=1.
    Retorna array NxM (float). Espera ao menos 4 colunas: X,Y,Z,grupo.
    """
    try:
        data = np.genfromtxt(path, delimiter=',', dtype=float)
    except Exception as e:
        print(f"Erro ao ler {path}: {e}")
        sys.exit(1)
    # Detectar linha com NaN
    if data.ndim == 1:
        # Caso tenha apenas uma linha ou algo estranho
        # Se possui NaN, possivelmente header único
        if np.isnan(data).any():
            try:
                data = np.genfromtxt(path, delimiter=',', dtype=float, skip_header=1)
            except Exception as e:
                print(f"Erro ao ler (segunda tentativa) {path}: {e}")
                sys.exit(1)
    else:
        # múltiplas linhas; verificar primeira linha
        if np.isnan(data[0]).any():
            # pular header
            try:
                data = np.genfromtxt(path, delimiter=',', dtype=float, skip_header=1)
            except Exception as e:
                print(f"Erro ao ler (skip_header) {path}: {e}")
                sys.exit(1)
    if data is None or data.size == 0:
        print(f"Nenhum dado lido de {path}. Verifique o CSV.")
        sys.exit(1)
    # Verificar colunas
    if data.shape[1] < 4:
        print(f"Esperado ao menos 4 colunas em {path} (X,Y,Z,grupo). Encontrado: {data.shape[1]}")
        sys.exit(1)
    return data

def compute_distance_matrix(coords):
    """
    coords: array Nx3
    Retorna matriz NxN com distâncias Euclidianas 3D.
    """
    # Usar broadcasting para cálculo eficiente
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    return dist

def nearest_neighbor_route(origin, dist_matrix):
    """
    Heurística Nearest Neighbor: retorna rota como lista de índices (exclui origem no início, mas a rota
    é: origin -> route -> origin)
    """
    n = dist_matrix.shape[0]
    unvisited = set(range(n))
    unvisited.discard(origin)
    route = []
    current = origin
    while unvisited:
        # escolhe vizinho mais próximo
        next_city = min(unvisited, key=lambda city: dist_matrix[current, city])
        route.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    return route

def compute_route_cost(route, origin, dist_matrix):
    """
    route: lista de índices (exclui origem). Calcula custo total ida (origin->primeiro) + entre pontos + volta (último->origin).
    """
    if len(route) == 0:
        return 0.0
    cost = dist_matrix[origin, route[0]]
    for i in range(len(route)-1):
        cost += dist_matrix[route[i], route[i+1]]
    cost += dist_matrix[route[-1], origin]
    return cost

def tournament_selection(population, costs, k):
    """
    population: lista de indivíduos (cada um é lista de índices), costs: lista de custos correspondentes.
    Seleciona k indivíduos aleatórios e retorna o de menor custo.
    """
    selected_idx = random.sample(range(len(population)), k)
    best_idx = min(selected_idx, key=lambda idx: costs[idx])
    # Retornar uma cópia para segurança
    return population[best_idx][:]

def order_crossover(parent1, parent2):
    """
    Order Crossover (OX) para permutações:
    - Escolhe dois pontos de corte i<j aleatórios.
    - Copia o segmento parent1[i:j+1] para child1 na mesma posição; preenche restante com genes de parent2 na ordem, pulando duplicados.
    - Análogo para child2.
    parent1, parent2: listas de mesma dimensão.
    Retorna child1, child2 (listas).
    """
    size = len(parent1)
    # Escolher dois pontos distintos
    i, j = sorted(random.sample(range(size), 2))
    child1 = [None] * size
    child2 = [None] * size
    # Copiar segmento
    child1[i:j+1] = parent1[i:j+1]
    child2[i:j+1] = parent2[i:j+1]
    # Preencher child1 com genes de parent2
    fill_pos = (j + 1) % size
    idx2 = (j + 1) % size
    while None in child1:
        gene = parent2[idx2]
        if gene not in child1:
            child1[fill_pos] = gene
            fill_pos = (fill_pos + 1) % size
        idx2 = (idx2 + 1) % size
    # Preencher child2 com genes de parent1
    fill_pos = (j + 1) % size
    idx1 = (j + 1) % size
    while None in child2:
        gene = parent1[idx1]
        if gene not in child2:
            child2[fill_pos] = gene
            fill_pos = (fill_pos + 1) % size
        idx1 = (idx1 + 1) % size
    return child1, child2

def mutate_swap(individual):
    """
    Swap mutation: escolhe duas posições aleatórias e troca.
    individual: lista de índices. Modifica in-place.
    """
    a, b = random.sample(range(len(individual)), 2)
    individual[a], individual[b] = individual[b], individual[a]

def run_ga(origin, dist_matrix,
           pop_size=200, max_gens=5000,
           tournament_k=3, crossover_prob=0.9,
           mutation_prob=0.05,
           elitism=True, elite_size=1,
           no_improve_window=250,
           use_accept_criterion=True, tol=0.05,
           verbose=False):
    """
    Executa GA básico:
    - origin: índice do ponto de origem.
    - dist_matrix: matriz NxN.
    - pop_size, max_gens: parâmetros de população e gerações.
    - tournament_k: tamanho do torneio.
    - crossover_prob: probabilidade de aplicar crossover.
    - mutation_prob: probabilidade de mutação por indivíduo.
    - elitism: bool; elite_size: número de indivíduos de elite.
    - no_improve_window: janela para parada sem melhora.
    - use_accept_criterion: se True, para caso atinja custo <= custo_NN * (1 - tol).
    - tol: tolerância relativa (e.g. 0.05 para 5% de melhora em relação a heurística NN).
    Retorna: best_ind, best_cost, history (lista de best_cost por geração), gen_atual, flag_aceitavel (bool).
    """
    n = dist_matrix.shape[0]
    # Base de índices de pontos exceto origem
    base = [i for i in range(n) if i != origin]
    size_perm = len(base)
    # Inicializar população aleatória
    population = [random.sample(base, size_perm) for _ in range(pop_size)]
    # Calcular custos iniciais
    costs = [compute_route_cost(ind, origin, dist_matrix) for ind in population]
    # Melhor global
    best_cost = min(costs)
    best_ind = population[costs.index(best_cost)][:]
    history = [best_cost]
    gens_no_improve = 0
    # Heurística NN para critério aceitável, se ativado
    cost_ref = None
    if use_accept_criterion:
        nn_route = nearest_neighbor_route(origin, dist_matrix)
        cost_ref = compute_route_cost(nn_route, origin, dist_matrix)
        if verbose:
            print(f"[GA] Custo referência NN = {cost_ref:.4f}, tol = {tol*100:.1f}%")
    # Loop de gerações
    for gen in range(1, max_gens+1):
        new_pop = []
        # Elitismo: copiar top elite_size indivíduos
        if elitism and elite_size > 0:
            sorted_idx = sorted(range(len(population)), key=lambda idx: costs[idx])
            for idx in sorted_idx[:elite_size]:
                # copia para nova pop
                new_pop.append(population[idx][:])
        # Gerar descendentes até completar
        while len(new_pop) < pop_size:
            # Seleção
            p1 = tournament_selection(population, costs, tournament_k)
            p2 = tournament_selection(population, costs, tournament_k)
            # Crossover
            if random.random() < crossover_prob:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            # Mutação swap
            if random.random() < mutation_prob:
                mutate_swap(c1)
            if random.random() < mutation_prob and len(new_pop) + 1 < pop_size:
                mutate_swap(c2)
            # Adicionar
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        population = new_pop
        # Calcular custos
        costs = [compute_route_cost(ind, origin, dist_matrix) for ind in population]
        current_best = min(costs)
        if current_best < best_cost - 1e-12:
            best_cost = current_best
            best_ind = population[costs.index(best_cost)][:]
            gens_no_improve = 0
        else:
            gens_no_improve += 1
        history.append(best_cost)
        # Logs periódicos
        if verbose and gen % 50 == 0:
            print(f"[GA] Geração {gen}: melhor custo = {best_cost:.4f}")
        # Critério de parada: solução aceitável
        if use_accept_criterion and cost_ref is not None:
            if best_cost <= cost_ref * (1 - tol):
                if verbose:
                    print(f"[GA] Solução aceitável atingida na geração {gen}, custo {best_cost:.4f}")
                return best_ind, best_cost, history, gen, True
        # Parada sem melhora
        if gens_no_improve >= no_improve_window:
            if verbose:
                print(f"[GA] Parada sem melhora em {no_improve_window} gerações (geração {gen}), custo {best_cost:.4f}")
            return best_ind, best_cost, history, gen, False
    # Se chegou aqui, atingiu max_gens
    if verbose:
        print(f"[GA] Atingiu max_gens ({max_gens}), melhor custo = {best_cost:.4f}")
    return best_ind, best_cost, history, max_gens, False

def plot_convergence(history, show_plot=True, save_path=None):
    """
    Plota curva de convergência: melhor custo por geração.
    """
    plt.figure(figsize=(8,5))
    plt.plot(history, label='Melhor custo')
    plt.xlabel('Geração')
    plt.ylabel('Custo')
    plt.title('Curva de Convergência do GA')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

def plot_route_3d(coords, route, origin_idx, show_plot=True, save_path=None):
    """
    Plota rota em 3D: coords é array Nx3, route é lista de índices incluindo origin no início e no fim.
    """
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    # plot pontos
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], color='gray', s=10, alpha=0.5)
    # plot linha da rota
    route_coords = coords[route]
    ax.plot(route_coords[:,0], route_coords[:,1], route_coords[:,2], color='red', linewidth=2)
    # destacar origem
    ax.scatter(coords[origin_idx,0], coords[origin_idx,1], coords[origin_idx,2],
               color='blue', s=50, label='Origem')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Rota Final do GA em 3D')
    ax.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='GA básico para Problema do Caixeiro Viajante 3D')
    parser.add_argument('--data_path', type=str, default='data/CaixeiroGruposGA.csv',
                        help='Caminho para o CSV de pontos (colunas X,Y,Z,grupo)')
    parser.add_argument('--origin_idx', type=int, default=0,
                        help='Índice do ponto de origem (linha no CSV, começando em 0)')
    # Parâmetros do GA
    parser.add_argument('--pop_size', type=int, default=200, help='Tamanho da população')
    parser.add_argument('--max_gens', type=int, default=5000, help='Número máximo de gerações')
    parser.add_argument('--tournament_k', type=int, default=3, help='Tamanho do torneio')
    parser.add_argument('--crossover_prob', type=float, default=0.9, help='Probabilidade de crossover')
    parser.add_argument('--mutation_prob', type=float, default=0.01, help='Probabilidade de mutação (swap) por indivíduo')
    parser.add_argument('--elitism', type=lambda x: (str(x).lower() in ['true','1','yes']), default=True,
                        help='Usar elitismo? (True/False)')
    parser.add_argument('--elite_size', type=int, default=1, help='Número de indivíduos de elite a manter')
    parser.add_argument('--no_improve_window', type=int, default=250, help='Gerações de parada sem melhora')
    parser.add_argument('--use_accept_criterion', type=lambda x: (str(x).lower() in ['true','1','yes']),
                        default=True, help='Usar critério de solução aceitável via heurística NN? (True/False)')
    parser.add_argument('--tolerance', type=float, default=0.05,
                        help='Tolerância relativa para solução aceitável (e.g. 0.05 = 5% de melhora sobre NN)')
    parser.add_argument('--save_plots', type=lambda x: (str(x).lower() in ['true','1','yes']), default=True,
                        help='Salvar plots em disco? (True/False)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Pasta para salvar plots (se save_plots=True)')
    parser.add_argument('--verbose', type=lambda x: (str(x).lower() in ['true','1','yes']), default=True,
                        help='Mostrar logs detalhados do GA')
    args = parser.parse_args()

    # Carregar CSV
    data = load_csv_numpy(args.data_path)
    # Separar colunas
    coords = data[:, 0:3]
    grupo = data[:, 3].astype(int)
    n_points = coords.shape[0]
    print(f"[Main] Total de pontos carregados: {n_points}")
    # Contagem por grupo
    unique, counts = np.unique(grupo, return_counts=True)
    print("[Main] Contagem de pontos por grupo:")
    for u, c in zip(unique, counts):
        print(f"  Grupo {u}: {c} pontos")
    # Validar origin_idx
    origin = args.origin_idx
    if origin < 0 or origin >= n_points:
        print(f"[Erro] origin_idx inválido: {origin}. Deve estar em [0, {n_points-1}].")
        sys.exit(1)
    print(f"[Main] Origem definida: índice {origin}, coordenadas {coords[origin]}")

    # Matriz de distâncias
    dist_matrix = compute_distance_matrix(coords)

    # Heurística Nearest Neighbor para referência
    if args.use_accept_criterion:
        nn_route = nearest_neighbor_route(origin, dist_matrix)
        nn_cost = compute_route_cost(nn_route, origin, dist_matrix)
        print(f"[Main] Heurística NN: custo = {nn_cost:.4f}")
    else:
        nn_cost = None

    # Executar GA
    best_ind, best_cost, history, gen_atual, accept_flag = run_ga(
        origin, dist_matrix,
        pop_size=args.pop_size,
        max_gens=args.max_gens,
        tournament_k=args.tournament_k,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        elitism=args.elitism,
        elite_size=args.elite_size,
        no_improve_window=args.no_improve_window,
        use_accept_criterion=args.use_accept_criterion,
        tol=args.tolerance,
        verbose=args.verbose
    )
    print(f"[Main] Resultado GA: melhor custo = {best_cost:.4f}, geração de parada = {gen_atual}, aceitável = {accept_flag}")

    # Preparar pasta de saída
    if args.save_plots:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = None

    # Plot de convergência
    conv_path = None
    if out_dir:
        conv_path = os.path.join(out_dir, 'convergence.png')
    plot_convergence(history, show_plot=True, save_path=conv_path)
    if conv_path:
        print(f"[Main] Curva de convergência salva em: {conv_path}")

    # Plot da rota final em 3D
    route_full = [origin] + best_ind + [origin]
    route_path = None
    if out_dir:
        route_path = os.path.join(out_dir, 'route_3d.png')
    plot_route_3d(coords, route_full, origin, show_plot=True, save_path=route_path)
    if route_path:
        print(f"[Main] Rota 3D salva em: {route_path}")

    # Fim
    print("[Main] Execução concluída.")

if __name__ == '__main__':
    main()
