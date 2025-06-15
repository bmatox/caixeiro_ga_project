import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import os
import sys

def load_csv_numpy(path):
    try:
        data = np.genfromtxt(path, delimiter=',', dtype=float)
    except Exception as e:
        print(f"Erro ao ler {path}: {e}")
        sys.exit(1)
    if data.ndim == 1:
        if np.isnan(data).any():
            try:
                data = np.genfromtxt(path, delimiter=',', dtype=float, skip_header=1)
            except Exception as e:
                print(f"Erro ao ler (segunda tentativa) {path}: {e}")
                sys.exit(1)
    else:
        if np.isnan(data[0]).any():
            try:
                data = np.genfromtxt(path, delimiter=',', dtype=float, skip_header=1)
            except Exception as e:
                print(f"Erro ao ler (skip_header) {path}: {e}")
                sys.exit(1)
    if data is None or data.size == 0:
        print(f"Nenhum dado lido de {path}. Verifique o CSV.")
        sys.exit(1)
    if data.shape[1] < 4:
        print(f"Esperado ao menos 4 colunas em {path} (X,Y,Z,grupo). Encontrado: {data.shape[1]}")
        sys.exit(1)
    return data

def compute_distance_matrix(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    return dist

def nearest_neighbor_route(origin, dist_matrix):
    n = dist_matrix.shape[0]
    unvisited = set(range(n))
    unvisited.discard(origin)
    route = []
    current = origin
    while unvisited:
        next_city = min(unvisited, key=lambda city: dist_matrix[current, city])
        route.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    return route

def compute_route_cost(route, origin, dist_matrix):
    if len(route) == 0:
        return 0.0
    cost = dist_matrix[origin, route[0]]
    for i in range(len(route)-1):
        cost += dist_matrix[route[i], route[i+1]]
    cost += dist_matrix[route[-1], origin]
    return cost

def tournament_selection(population, costs, k):
    selected_idx = random.sample(range(len(population)), k)
    best_idx = min(selected_idx, key=lambda idx: costs[idx])
    return population[best_idx][:]  # cópia

def order_crossover(parent1, parent2):
    size = len(parent1)
    i, j = sorted(random.sample(range(size), 2))
    child1 = [None] * size
    child2 = [None] * size
    child1[i:j+1] = parent1[i:j+1]
    child2[i:j+1] = parent2[i:j+1]
    fill_pos = (j + 1) % size
    idx2 = (j + 1) % size
    while None in child1:
        gene = parent2[idx2]
        if gene not in child1:
            child1[fill_pos] = gene
            fill_pos = (fill_pos + 1) % size
        idx2 = (idx2 + 1) % size
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
    a, b = random.sample(range(len(individual)), 2)
    individual[a], individual[b] = individual[b], individual[a]

def run_ga(origin, dist_matrix,
           pop_size=100, max_gens=500,
           tournament_k=3, crossover_prob=0.9,
           mutation_prob=0.01,
           elitism=True, elite_size=1,
           no_improve_window=50,
           use_accept_criterion=True, tol=0.05,
           verbose=False):
    n = dist_matrix.shape[0]
    base = [i for i in range(n) if i != origin]
    size_perm = len(base)
    population = [random.sample(base, size_perm) for _ in range(pop_size)]
    costs = [compute_route_cost(ind, origin, dist_matrix) for ind in population]
    best_cost = min(costs)
    best_ind = population[costs.index(best_cost)][:]
    history = [best_cost]
    gens_no_improve = 0
    cost_ref = None
    if use_accept_criterion:
        nn_route = nearest_neighbor_route(origin, dist_matrix)
        cost_ref = compute_route_cost(nn_route, origin, dist_matrix)
        if verbose:
            print(f"[GA] Custo referência NN = {cost_ref:.4f}, tol = {tol*100:.1f}%")
    for gen in range(1, max_gens+1):
        new_pop = []
        if elitism and elite_size > 0:
            sorted_idx = sorted(range(len(population)), key=lambda idx: costs[idx])
            for idx in sorted_idx[:elite_size]:
                new_pop.append(population[idx][:])
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, costs, tournament_k)
            p2 = tournament_selection(population, costs, tournament_k)
            if random.random() < crossover_prob:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            if random.random() < mutation_prob:
                mutate_swap(c1)
            if random.random() < mutation_prob and len(new_pop) + 1 < pop_size:
                mutate_swap(c2)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        population = new_pop
        costs = [compute_route_cost(ind, origin, dist_matrix) for ind in population]
        current_best = min(costs)
        if current_best < best_cost - 1e-12:
            best_cost = current_best
            best_ind = population[costs.index(best_cost)][:]
            gens_no_improve = 0
        else:
            gens_no_improve += 1
        history.append(best_cost)
        if verbose and gen % 50 == 0:
            print(f"[GA] Geração {gen}: melhor custo = {best_cost:.4f}")
        if use_accept_criterion and cost_ref is not None:
            if best_cost <= cost_ref * (1 - tol):
                if verbose:
                    print(f"[GA] Solução aceitável atingida na geração {gen}, custo {best_cost:.4f}")
                return best_ind, best_cost, history, gen, True
        if gens_no_improve >= no_improve_window:
            if verbose:
                print(f"[GA] Parada sem melhora em {no_improve_window} gerações (geração {gen}), custo {best_cost:.4f}")
            return best_ind, best_cost, history, gen, False
    if verbose:
        print(f"[GA] Atingiu max_gens ({max_gens}), melhor custo = {best_cost:.4f}")
    return best_ind, best_cost, history, max_gens, False

def plot_convergence(history, show_plot=True, save_path=None):
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

def plot_route_3d (coords, groups, route, origin_idx, show_plot=True, save_path=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    group_colors = {1: 'red', 2: 'green', 3: 'blue', 4: 'purple'}
    group_labels = {1: 'Região 1', 2: 'Região 2', 3: 'Região 3', 4: 'Região 4'}

    for group_id in np.unique(groups):
        if group_id == 0: continue

        idx_group = np.where(groups == group_id)[0]
        ax.scatter(coords[idx_group, 0], coords[idx_group, 1], coords[idx_group, 2],
                   color=group_colors.get(group_id, 'gray'),
                   s=40,
                   label=group_labels.get(group_id, f'Grupo {group_id}'))

    ax.scatter(coords[origin_idx, 0], coords[origin_idx, 1], coords[origin_idx, 2],
               color='black',  
               s=250,          
               marker='*',    
               edgecolor='white',
               linewidth=1,
               label=f'Origem (Ponto {origin_idx})')
    route_coords = coords[route]
    ax.plot(route_coords[:, 0], route_coords[:, 1], route_coords[:, 2],
            color='cyan', linewidth=1.5, alpha=0.8, label='Rota Otimizada')

    ax.set_xlabel('Eixo X')
    ax.set_ylabel('Eixo Y')
    ax.set_zlabel('Eixo Z')
    ax.set_title('Rota Final Otimizada do Drone por Região')
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

def main():
    SEED = 30
    random.seed(SEED)
    np.random.seed(SEED)
    parser = argparse.ArgumentParser(description='GA básico para Problema do Caixeiro Viajante 3D')
    parser.add_argument('--data_path', type=str, default='caixeiro_ga_project\\data\\CaixeiroGruposGA.csv',
                        help='Caminho para o CSV de pontos (colunas X,Y,Z,grupo)')
    parser.add_argument('--origin_idx', type=int, default=0,
                        help='Índice do ponto de origem (linha no CSV, começando em 0)')
    parser.add_argument('--n_per_region', type=int, default=None,
                        help='Número de pontos a selecionar por região (30 <= n <= 60). Se None, usa todos do CSV.')
    parser.add_argument('--pop_size', type=int, default=250, help='Tamanho da população') #melhor tamanho de pop encontrado foi 250
    parser.add_argument('--max_gens', type=int, default=8000, help='Número máximo de gerações')
    parser.add_argument('--tournament_k', type=int, default=3, help='Tamanho do torneio') #3 é o melhor torneio, melhor q 2 e 4.
    parser.add_argument('--crossover_prob', type=float, default=0.92, help='Probabilidade de crossover') #2188
    parser.add_argument('--mutation_prob', type=float, default=0.01, help='Probabilidade de mutação (swap) por indivíduo')
    parser.add_argument('--elitism', type=lambda x: (str(x).lower() in ['true','1','yes']), default=True,
                        help='Usar elitismo? (True/False)')
    parser.add_argument('--elite_size', type=int, default=1, help='Número de indivíduos de elite a manter')
    parser.add_argument('--no_improve_window', type=int, default=350, help='Gerações de parada sem melhora') #350 um resultado um pouco pior que 450
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
    coords_all = data[:, 0:3]
    grupo_all = data[:, 3].astype(int)
    n_total = coords_all.shape[0]
    print(f"[Main] Total de pontos carregados: {n_total}")
    unique, counts = np.unique(grupo_all, return_counts=True)
    print("[Main] Contagem de pontos por grupo (arquivo original):")
    for u, c in zip(unique, counts):
        print(f"  Grupo {u}: {c} pontos")

    origin_orig = args.origin_idx
    if origin_orig < 0 or origin_orig >= n_total:
        print(f"[Erro] origin_idx inválido: {origin_orig}. Deve estar em [0, {n_total-1}].")
        sys.exit(1)
    print(f"[Main] Origem escolhida (no CSV original): índice {origin_orig}, coordenadas {coords_all[origin_orig]}")

    if args.n_per_region is not None:
        n_pr = args.n_per_region
        if not (30 <= n_pr <= 60):
            print(f"[Erro] n_per_region deve estar entre 30 e 60. Recebido: {n_pr}")
            sys.exit(1)
        selected_indices = []
        origin_group = grupo_all[origin_orig]
        for grp in unique:
            inds = np.where(grupo_all == grp)[0].tolist()
            if len(inds) < n_pr:
                print(f"[Erro] Grupo {grp} tem apenas {len(inds)} pontos no CSV, menor que n_per_region={n_pr}.")
                sys.exit(1)
            if grp == origin_group:
                others = [i for i in inds if i != origin_orig]
                sampled = random.sample(others, n_pr - 1)
                sampled.append(origin_orig)
            else:
                sampled = random.sample(inds, n_pr)
            selected_indices.extend(sampled)
        selected_indices = sorted(selected_indices)  
        coords = coords_all[selected_indices]
        grupo = grupo_all[selected_indices]
        origin_idx = selected_indices.index(origin_orig)
        print(f"[Main] Após amostragem, total de pontos usado: {coords.shape[0]} ({n_pr} por grupo).")
        print(f"[Main] Novo índice da origem no conjunto amostrado: {origin_idx}")
    else:
        coords = coords_all
        grupo = grupo_all
        origin_idx = origin_orig
        print("[Main] Usando todos os pontos do CSV (nenhuma amostragem por região).")

    n_points = coords.shape[0]
    unique2, counts2 = np.unique(grupo, return_counts=True)
    print("[Main] Contagem de pontos por grupo (conjunto usado):")
    for u, c in zip(unique2, counts2):
        print(f"  Grupo {u}: {c} pontos")
    print(f"[Main] Total de pontos usados para o GA: {n_points}")
    print(f"[Main] Origem efetiva: índice {origin_idx}, coordenadas {coords[origin_idx]}")

    dist_matrix = compute_distance_matrix(coords)

    if args.use_accept_criterion:
        nn_route = nearest_neighbor_route(origin_idx, dist_matrix)
        nn_cost = compute_route_cost(nn_route, origin_idx, dist_matrix)
        print(f"[Main] Heurística NN: custo = {nn_cost:.4f}")
    else:
        nn_cost = None

    best_ind, best_cost, history, gen_atual, accept_flag = run_ga(
        origin_idx, dist_matrix,
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
    if args.save_plots:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = None
    conv_path = None
    if out_dir:
        conv_path = os.path.join(out_dir, 'convergence.png')
    plot_convergence(history, show_plot=True, save_path=conv_path)
    if conv_path:
        print(f"[Main] Curva de convergência salva em: {conv_path}")

    route_full = [origin_idx] + best_ind + [origin_idx]
    route_path = None
    if out_dir:
        route_path = os.path.join(out_dir, 'route_3d.png')
    plot_route_3d(coords, grupo, route_full, origin_idx, show_plot=True, save_path=route_path)
    if route_path:
        print(f"[Main] Rota 3D salva em: {route_path}")

    print("[Main] Execução concluída.")

if __name__ == '__main__':
    main()