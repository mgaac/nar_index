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

    def __call__(self, connection_matrix, node_embeddings):

        mask = mx.random.bernoulli(self.dropout_prob, connection_matrix.shape)
        connection_matrix = connection_matrix * mask

        edge_weights = mx.expand_dims(connection_matrix[2], axis=0)
        edge_weights = mx.expand_dims(edge_weights, axis=-1)

        connection_matrix = connection_matrix[:2].astype(mx.int32)

        node_embeddings = self.dropout(node_embeddings)

        source_idx = connection_matrix[self.source_idx]
        target_idx = connection_matrix[self.target_idx]

        source_embeddings = node_embeddings @ self.source_message_fn
        target_embeddings = node_embeddings @ self.target_message_fn

        filtered_source_embeddings = mx.take(source_embeddings, source_idx, axis=1)
        filtered_target_embeddings = mx.take(target_embeddings, target_idx, axis=1)

        message = filtered_source_embeddings + filtered_target_embeddings

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
    def __init__(self, num_nodes: int,embedding_dim: int, dim_proj: int, dropout_prob: float, skip_connections: bool, aggregation_fn: Enum, num_mp_layers: int):
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

    def __call__(self, data):
        node_embeddings, connection_matrix = data

        assert node_embeddings.shape[1] == self.embedding_dim, f'Incorrect node embedding size. Expected {self.embedding_dim}, got {node_embeddings.shape[1]}'

        for mp_layer in self.mp_layer:
            node_embeddings = mp_layer(connection_matrix, node_embeddings)

        return node_embeddings
    
class nge(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int, dim_proj: int, dropout_prob: float, skip_connections: bool, aggregation_fn: Enum, num_mp_layers: int):
        super(nge, self).__init__()
        
        self.parallel_encoder = nn.Linear(num_nodes, embedding_dim)
        self.sequential_encoder = nn.Linear(num_nodes, embedding_dim)

        self.parallel_decoder = nn.Linear(embedding_dim, 3)
        self.sequential_decoder = nn.Linear(embedding_dim, 2)
    
        self.parallel_termination_fn = nn.Sequential(nn.Linear(2 * embedding_dim, 1), nn.ReLU())
        self.sequential_termination_fn = nn.Sequential(nn.Linear(2 * embedding_dim, 1), nn.ReLU())

        self.processor = mpnn(num_nodes, embedding_dim, dim_proj, dropout_prob, skip_connections, aggregation_fn, num_mp_layers)

    def __call__(self, data, task):
        node_embeddings, connection_matrix = data

        if task == task.PARALLEL_ALGORIHTM:
            node_embeddings = self.parallel_encoder(node_embeddings)
            new_node_embeddings = self.processor((node_embeddings, connection_matrix))
            output = self.parallel_decoder(new_node_embeddings)
        elif task == task.SEQUENTIAL_ALGORITHM:
            node_embeddings = self.sequential_encoder(node_embeddings)
            new_node_embeddings = self.processor((node_embeddings, connection_matrix))
            output = self.sequential_decoder(new_node_embeddings)

        avg_node_embeddings = mx.mean(new_node_embeddings, axis=0)

        if task == task.PARALLEL_ALGORIHTM:
            termination_prob = self.parallel_termination_fn(mx.concatenate([new_node_embeddings, avg_node_embeddings], axis=1))
        elif task == task.SEQUENTIAL_ALGORITHM:
            termination_prob = self.sequential_termination_fn(mx.concatenate([new_node_embeddings, avg_node_embeddings], axis=1))

        return new_node_embeddings, output, termination_prob

#Dataset generation
train_graphs = load_graphs('train_graphs.pkl')
val_graphs = load_graphs('val_graphs.pkl')
test_graphs_20 = load_graphs('test_graphs_20.pkl')
test_graphs_50 = load_graphs('test_graphs_50.pkl')
test_graphs_100 = load_graphs('test_graphs_100.pkl')


model_config = {
    'num_nodes': 20,
    'embedding_dim': 32,
    'dim_proj': 1,
    'dropout_prob': 0.5,
    'skip_connections': True,
    'aggregation_fn': aggregation_fn.MAX,
    'num_mp_layers': 1
}

hyper_params = {
    'learning_rate': 0.001,
    'num_steps': 1000
}

model = nge(**model_config)
optimizer = optim.Adam(learning_rate=hyper_params["learning_rate"])


def sequential_loss_fn(model, input, regular_target, termination_target):
    new_node_embeddings, output, termination_prob = model(input, task.SEQUENTIAL_ALGORITHM)
    state, predesecor = output[:,0], output[:,1]

    reachability_target, predesecor_target = regular_target

    state_loss = nn.losses.binary_cross_entropy(state, reachability_target)
    pred_loss = nn.losses.cross_entropy(predesecor, predesecor_target)

    termination_loss = nn.losses.binary_cross_entropy(termination_prob, termination_target)

    return (state_loss, pred_loss, termination_loss), new_node_embeddings

def parallel_loss_fn(model, input, regular_target, termination_target):
    new_node_embeddings, output, termination_prob = model(input, task.PARALLEL_ALGORIHTM)
    bf_state, bfs_predesecor, bfs_distance = output[:,0], output[:,1], output[:,2]

    reachability_target, predesecor_target, distance_target = regular_target

    bf_state_loss = nn.losses.binary_cross_entropy(bf_state, reachability_target)
    bfs_predesecor_loss = nn.losses.cross_entropy(bfs_predesecor, predesecor_target)
    bfs_distance_loss = nn.losses.mse_loss(bfs_distance, distance_target)

    termination_loss = nn.losses.binary_cross_entropy(termination_prob, termination_target)

    return (bf_state_loss, bfs_predesecor_loss, bfs_distance_loss, termination_loss), new_node_embeddings