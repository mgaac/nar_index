import networkx as nx
import mlx.core as mx
import numpy as np
import random
import pickle
from tqdm import tqdm
from collections import defaultdict

from enum import Enum

class task(Enum):
    PARALLEL_ALGORIHTM = 0
    SEQUENTIAL_ALGORITHM = 1

def generate_graphs(num_nodes):
    graphs = []

    # 1. Ladder graph
    G_ladder = nx.ladder_graph(num_nodes // 2)
    graphs.append(G_ladder)

    # 2. 2D grid graph
    side = int(np.sqrt(num_nodes))
    G_grid = nx.grid_2d_graph(side, side)
    G_grid = nx.convert_node_labels_to_integers(G_grid)
    graphs.append(G_grid)

    # 3. Random tree from Prüfer sequence
    G_tree = nx.random_spanning_tree(nx.complete_graph(num_nodes))
    graphs.append(G_tree)

    # 4. Erdős-Rényi graph
    p_er = min(np.log2(num_nodes) / num_nodes, 0.5)
    G_er = nx.erdos_renyi_graph(num_nodes, p_er)
    graphs.append(G_er)

    # 5. Barabási-Albert graph (attach 4 or 5 edges)
    m_ba = random.choice([4, 5])
    G_ba = nx.barabasi_albert_graph(num_nodes, m_ba)
    graphs.append(G_ba)

    # 6. 4-Community graph
    community_size = num_nodes // 4
    communities = [nx.erdos_renyi_graph(community_size, 0.7) for _ in range(4)]
    G_4comm = nx.disjoint_union_all(communities)
    # Add inter-community edges
    nodes = list(G_4comm.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if G_4comm.nodes[nodes[i]].get('block', i // community_size) != G_4comm.nodes[nodes[j]].get('block', j // community_size):
                if random.random() < 0.01:
                    G_4comm.add_edge(nodes[i], nodes[j])
    graphs.append(G_4comm)

    # 7. 4-Caveman graph
    G_caveman = nx.caveman_graph(4, community_size)
    # Remove intra-clique edges with probability 0.7
    to_remove = []
    for u, v in G_caveman.edges():
        if u // community_size == v // community_size and random.random() < 0.7:
            to_remove.append((u, v))
    G_caveman.remove_edges_from(to_remove)
    # Add shortcut edges
    num_shortcuts = int(0.025 * num_nodes)
    for _ in range(num_shortcuts):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u // community_size != v // community_size:
            G_caveman.add_edge(u, v)
    graphs.append(G_caveman)

    # Add self-loops and edge weights to all graphs
    for G in graphs:
        for n in G.nodes():
            G.add_edge(n, n)  # self-loop
        for u, v in G.edges():
            G[u][v]['weight'] = random.uniform(0.2, 1.0)

    return graphs

def edge_list_to_connection_matrix(edge_list):
    # Create source and target indices arrays
    sources = []
    targets = []
    weights = []
    
    for u, v, w in edge_list:
        sources.append(u)
        targets.append(v)
        weights.append(w)
    
    # Stack into a 3xN matrix where rows are [sources, targets, weights]
    return mx.array([sources, targets, weights])

def save_graphs(graphs, filename):
    edge_lists = []
    connection_matrices = []
    targets = []
    
    for G in graphs:
        # Create edge list with weights
        edge_list = []
        for u, v in G.edges():
            weight = G[u][v]['weight']
            edge_list.append((u, v, weight))
            edge_list.append((v, u, weight))  # Add reverse edge
            
        edge_lists.append(edge_list)
        connection_matrices.append(edge_list_to_connection_matrix(edge_list))
        # Generate targets for both tasks
        targets.append({
            'parallel': generate_targets(edge_list, 0, task.PARALLEL_ALGORIHTM),
            'sequential': generate_targets(edge_list, 0, task.SEQUENTIAL_ALGORITHM)
        })
    
    # Save edge lists, connection matrices, and targets
    with open(filename, 'wb') as f:
        pickle.dump({
            'edge_lists': edge_lists,
            'connection_matrices': connection_matrices,
            'targets': targets
        }, f)

def load_graphs(filename):
    with open('utils/datasets/nega_custom/data/' + filename, 'rb') as f:
        return pickle.load(f)
    
def bfs_edge_list(edges, start):
    nodes = set([node for edge in edges for node in edge[:2]])
    state = {node: 1 if node == start else 0 for node in nodes}
    state_history = [state.copy()]
    for _ in range(len(nodes) - 1):
        new_state = state.copy()
        for u, v, _ in edges:
            if state[u] == 1:
                new_state[v] = 1
            if state[v] == 1:
                new_state[u] = 1
        state = new_state
        state_history.append(state.copy())
    return state_history

def calculate_stable_infinity(edges, start):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    lengths = nx.single_source_dijkstra_path_length(G, source=start, weight='weight')
    return max(lengths.values()) + 1

def bellman_ford_edge_list(edges, start):
    nodes = set([node for edge in edges for node in edge[:2]])
    INFINITY = calculate_stable_infinity(edges, start)
    distance = {node: 0 if node == start else INFINITY for node in nodes}
    predecessor = {node: node if node == start else distance[node] for node in nodes}
    history = [{'distance': distance.copy(), 'predecessor': predecessor.copy()}]
    for _ in range(len(nodes) - 1):
        for u, v, w in edges:
            if distance[u] != INFINITY and distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
                predecessor[v] = u
        history.append({'distance': distance.copy(), 'predecessor': predecessor.copy()})
    return history

def prim_edge_list(edges, start):
    nodes = defaultdict(list)
    for u, v, w in edges:
        nodes[u].append((v, w))
        nodes[v].append((u, w))
    INFINITY = calculate_stable_infinity(edges, start)
    # Initialize MST state and distance to MST
    state = {node: 0 for node in nodes}
    distance = {node: INFINITY for node in nodes}
    state[start] = 1
    visited = {start}
    # Initialize distances from the start node
    for v, w in nodes[start]:
        distance[v] = w
    distance[start] = 0
    # Record initial history
    history = [{'state': state.copy(), 'distance': distance.copy()}]
    # Build MST
    while len(visited) < len(nodes):
        # Select unvisited node with smallest distance to current MST
        u = min((node for node in nodes if node not in visited), key=lambda x: distance[x])
        if distance[u] == INFINITY:
            break
        visited.add(u)
        state[u] = 1
        # Update neighbor distances
        for v, w in nodes[u]:
            if v not in visited and w < distance[v]:
                distance[v] = w
        # Record history after adding this node
        history.append({'state': state.copy(), 'distance': distance.copy()})
    return history

def generate_targets(graph, start, task):
    if task == task.PARALLEL_ALGORIHTM:
        bfs = bfs_edge_list(graph, start)
        bf = bellman_ford_edge_list(graph, start)
        bf_dist = mx.array([list(h['distance'].values()) for h in bf])
        bf_pred = mx.array([list(h['predecessor'].values()) for h in bf])
        bf = mx.concatenate([bf_pred, bf_dist], axis=0)
        bfs = mx.array([list(h.values()) for h in bfs])

        return mx.concatenate([bfs, bf], axis=0)

    elif task == task.SEQUENTIAL_ALGORITHM:
        prim = prim_edge_list(graph, start)
        prim_dist = mx.array([list(h['distance'].values()) for h in prim])
        prim_state = mx.array([list(h['state'].values()) for h in prim])
        prim = mx.concatenate([prim_state, prim_dist], axis=0)
        prim = mx.array(prim)

        return prim
    
if __name__ == "__main__":
    # Generate graphs with progress bars
    train_graphs_nested = [generate_graphs(20) for _ in tqdm(range(100), desc="Generating train graphs", leave=False)]
    val_graphs_nested = [generate_graphs(20) for _ in tqdm(range(5), desc="Generating validation graphs", leave=False)]
    test_graphs_20_nested = [generate_graphs(20) for _ in tqdm(range(5), desc="Generating test graphs (20 nodes)", leave=False)]
    test_graphs_50_nested = [generate_graphs(50) for _ in tqdm(range(5), desc="Generating test graphs (50 nodes)", leave=False)]
    test_graphs_100_nested = [generate_graphs(100) for _ in tqdm(range(5), desc="Generating test graphs (100 nodes)", leave=False)]

    # Flatten the lists
    train_graphs = [graph for graphs in train_graphs_nested for graph in graphs]
    val_graphs = [graph for graphs in val_graphs_nested for graph in graphs]
    test_graphs_20 = [graph for graphs in test_graphs_20_nested for graph in graphs]
    test_graphs_50 = [graph for graphs in test_graphs_50_nested for graph in graphs]
    test_graphs_100 = [graph for graphs in test_graphs_100_nested for graph in graphs]

    print("Saving graphs...\n")

    save_graphs(train_graphs, 'train_graphs.pkl')
    save_graphs(val_graphs, 'val_graphs.pkl')
    save_graphs(test_graphs_20, 'test_graphs_20.pkl')
    save_graphs(test_graphs_50, 'test_graphs_50.pkl')
    save_graphs(test_graphs_100, 'test_graphs_100.pkl')