import mlx
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import networkx as nx
import numpy as np
import random

from enum import Enum
from utils.datasets.nega_custom.scripts.graph_generator import load_graphs
from collections import deque, defaultdict

class task(Enum):
    PARALLEL_ALGORIHTM=0
    SEQUENTIAL_ALGORITHM=1

# Model Architecture
class aggregation_fn(Enum):
    SUM = 1
    AVG = 2
    MIN = 4
    MAX = 5

class mp_layer(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int, dim_proj: int, dropout_prob: float, skip_connections: bool, aggregation_fn: Enum):
        super().__init__()

        self.source_idx = 0
        self.target_idx = 1

        self.dim_proj = dim_proj
        self.num_nodes = num_nodes
        self.dropout_prob = dropout_prob
        self.skip_connections = skip_connections
        self.aggregation_fn = aggregation_fn

        self.source_message_fn = mx.random.normal([1, embedding_dim, dim_proj])
        self.target_message_fn = mx.random.normal([1, embedding_dim, dim_proj])

        self.update_fn = nn.Linear(dim_proj, embedding_dim)

        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def __call__(self, connection_matrix, node_embeddings, edge_weights=None):

        mask = mx.random.bernoulli(self.dropout_prob, connection_matrix.shape)
        connection_matrix = connection_matrix * mask

        node_embeddings = self.dropout(node_embeddings)

        source_idx = connection_matrix[self.source_idx]
        target_idx = connection_matrix[self.target_idx]

        source_embeddings = node_embeddings @ self.source_message_fn
        target_embeddings = node_embeddings @ self.target_message_fn

        filtered_source_embeddings = mx.take(source_embeddings, source_idx, axis=1)
        filtered_target_embeddings = mx.take(target_embeddings, target_idx, axis=1)

        message = filtered_source_embeddings + filtered_target_embeddings

        if edge_weights is not None:
            message = message * edge_weights

        message = self.relu(message)

        agg_message = mx.zeros([self.num_nodes, self.dim_proj])

        if (self.aggregation_fn == aggregation_fn.SUM):
            agg_message = agg_message.at[target_idx].add(message)

        elif (self.aggregation_fn == aggregation_fn.AVG):
            agg_message = agg_message.at[target_idx].add(message)
            denominator = mx.zeros([self.num_nodes, 1]).at[target_idx].add(1)
            agg_message = agg_message /  mx.maximum(denominator, 1e-6)

        elif (self.aggregation_fn == aggregation_fn.MAX):
            agg_message = agg_message.at[target_idx].maximum(message)

        elif (self.aggregation_fn == aggregation_fn.MIN):
            agg_message = agg_message.at[target_idx].minimum(message)

        agg_message = self.dropout(agg_message)
        new_node_embeddings = self.update_fn(agg_message)
        new_node_embeddings = self.relu(new_node_embeddings)

        if (self.skip_connections):
            new_node_embeddings = new_node_embeddings + node_embeddings

        return new_node_embeddings

class mpnn(nn.Module):
    def __init__(self, num_nodes: int,embedding_dim: int, dim_proj: int, dropout_prob: float, skip_connections: bool, aggregation_fn: Enum, num_mp_layers: int, num_out_layers, num_classes: int):
        super(mpnn, self).__init__()

        self.embedding_dim = embedding_dim
        self.dim_proj = dim_proj
        self.num_nodes = num_nodes
        self.dropout_prob = dropout_prob
        self.skip_connections = skip_connections
        self.aggregation_function_fn = aggregation_fn

        self.mp_layer = [
            mp_layer(num_nodes, embedding_dim, dim_proj, dropout_prob, skip_connections, aggregation_fn)
            for _ in range(num_mp_layers)
        ]

        self.out_layer = [
            nn.Linear(embedding_dim, embedding_dim)
            for _ in range(num_out_layers)
        ] + [nn.Linear(embedding_dim, num_classes)]

    def __call__(self, data):
        node_embeddings, connection_matrix = data

        assert node_embeddings.shape[1] == self.embedding_dim, f'Incorrect node embedding size. Expected {self.embedding_dim}, got {node_embeddings.shape[1]}'

        for mp_layer in self.mp_layer:
            node_embeddings = mp_layer(connection_matrix, node_embeddings)

        for out_layer in self.out_layer:
            node_embeddings = out_layer(node_embeddings)

        return node_embeddings

#Dataset generation
train_graphs = load_graphs('train_graphs.pkl')
val_graphs = load_graphs('val_graphs.pkl')
test_graphs_20 = load_graphs('test_graphs_20.pkl')
test_graphs_50 = load_graphs('test_graphs_50.pkl')
test_graphs_100 = load_graphs('test_graphs_100.pkl')


model_config = {
    'num_nodes': 2708,
    'embedding_dim': 1433,
    'dim_proj': 8,
    'dropout_prob': 0.5,
    'skip_connections': True,
    'aggregation_fn': aggregation_fn.SUM,
    'num_mp_layers': 1,
    'num_out_layers': 1,
    'num_classes': 7
}

hyper_params = {
    'learning_rate': 0.001,
    'num_steps': 1000
}

encoder = nn.Linear(model_config["embedding_dim"], model_config["embedding_dim"])
decoder = nn.Linear(model_config["embedding_dim"], model_config["embedding_dim"])
processor = mpnn(**model_config)

mx.eval(encoder.parameters())
mx.eval(decoder.parameters())
mx.eval(processor.parameters()) 

optimizer = optim.Adam(learning_rate=hyper_params["learning_rate"])

def bfs_edge_list(edges, start):
    # Get all nodes
    nodes = set([node for edge in edges for node in edge[:2]])
    
    # Initialize with binary values as per paper
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
    
    # Initialize distances and predecessors
    distance = {node: 0 if node == start else INFINITY for node in nodes}
    predecessor = {node: node if node == start else distance[node] for node in nodes}
    
    history = [{'distance': distance.copy(), 'predecessor': predecessor.copy()}]
    
    for _ in range(len(nodes) - 1):
        for u, v, w in edges:
            if distance[u] != INFINITY and distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
                predecessor[v] = u
        
        history.append({
            'distance': distance.copy(),
            'predecessor': predecessor.copy()
        })
    
    return history

def prim_edge_list(edges, start):
    nodes = defaultdict(list)
    for u, v, w in edges:
        nodes[u].append((v, w))
        nodes[v].append((u, w))

    INFINITY = calculate_stable_infinity(edges, start)
    
    state = {node: 1 if node == start else 0 for node in nodes}
    distance = {node: 0 if node == start else INFINITY for node in nodes}

    history = [{'state': state.copy(), 'distance': distance.copy()}]
    
    queue = deque([start])
    visited = set([start])

    while queue:
        u = queue.popleft()
        next_node, _ = min(nodes[u], key=lambda x: x[1])
        queue.append(next_node)
        visited.add(next_node)

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

# Process each graph individually to reduce memory usage
num_graphs_to_process = 1 # Change as needed

parallel_targets = []
sequential_targets = []

for i, graph in enumerate(test_graphs_20[:num_graphs_to_process]):
    pt = generate_targets(graph, 0, task.PARALLEL_ALGORIHTM)
    st = generate_targets(graph, 0, task.SEQUENTIAL_ALGORITHM)
    parallel_targets.append(pt)
    sequential_targets.append(st)
    print(f"Graph {i}: parallel_target shape: {pt.shape}, sequential_target shape: {st.shape}")

# Repeat for other datasets as needed
# for i, graph in enumerate(val_graphs[:num_graphs_to_process]):
#     ...
# for i, graph in enumerate(test_graphs_20[:num_graphs_to_process]):
#     ...
# for i, graph in enumerate(test_graphs_50[:num_graphs_to_process]):
#     ...
# for i, graph in enumerate(test_graphs_100[:num_graphs_to_process]):
#     ...