import networkx as nx
import numpy as np
import random
import pickle
from tqdm import tqdm

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

def save_graphs(graphs, filename):
    edge_lists = []
    for G in graphs:
        edge_list = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
        edge_lists.append(edge_list)
    with open(filename, 'wb') as f:
        pickle.dump(edge_lists, f)

def load_graphs(filename):
    with open('utils/datasets/nega_custom/data/' + filename, 'rb') as f:
        return pickle.load(f)

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