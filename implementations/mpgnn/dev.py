import mlx.core as mx
import mlx.nn as nn

import mlx.optimizers as optim
from mlx.utils import tree_map
from functools import partial

from enum import Enum

import numpy as np
import pickle
import os

class aggregation_fn(Enum):
    SUM = 1
    AVG = 2
    MAX = 3
    MIN = 4

class mp_layer(nn.Module):
    def __init__(self, num_nodes: int, dim_proj: int, dropout_prob: float, skip_connections: bool, aggregation_fn: Enum):
        super().__init__()

        self.source_idx = 0
        self.target_idx = 1

        self.dim_proj = dim_proj
        self.num_nodes = num_nodes
        self.dropout_prob = dropout_prob
        self.skip_connections = skip_connections
        self.aggregation_function_fn = aggregation_fn

        self.source_message_fn = mx.random.normal([1, num_nodes, dim_proj])
        self.target_message_fn = mx.random.normal([1, num_nodes, dim_proj])

        self.update_fn = nn.Linear(2 * dim_proj, dim_proj)

        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def __call__(self, connection_matrix, node_embeddings):

        connection_matrix = self.dropout(connection_matrix)
        node_embeddings = self.dropout(node_embeddings)

        source_idx = connection_matrix[self.source_idx]
        target_idx = connection_matrix[self.target_idx]

        source_embeddings = self.source_message_fn @ node_embeddings
        target_embeddings = self.target_message_fn @ node_embeddings

        filtered_source_embeddings = mx.take(source_embeddings, source_idx)
        filtered_target_embeddings = mx.take(target_embeddings, target_idx)

        message = mx.concatenate(filtered_source_embeddings, filtered_target_embeddings, axis=1)

        agg_message = mx.zeros([self.num_nodes, self.dim_proj])

        if (aggregation_fn is aggregation_fn.SUM):
            agg_message = agg_message.at[target_idx].add(message)

        elif (aggregation_fn is aggregation_fn.AVG):
            agg_message = agg_message.at[target_idx].add(message)
            denominator = mx.zeros([self.num_nodes]).at[target_idx].add(1)
            agg_message = agg_message.at[target_idx].div(denominator)

        elif (aggregation_fn is aggregation_fn.MAX):
            agg_message = agg_message.at[target_idx].maximum(message)

        elif (aggregation_fn is aggregation_fn.MIN):
            agg_message = agg_message.at[target_idx].minimum(message)

        agg_message = self.dropout(agg_message)
        new_node_embeddings = self.update_fn(agg_message)
        new_node_embeddings = self.relu(new_node_embeddings)

        if (self.skip_connections):
            new_node_embeddings = new_node_embeddings + node_embeddings

        return new_node_embeddings


class mpnn(nn.Module):
    def __init__(self, num_nodes: int, dim_proj: int, dropout_prob: float, skip_connections: bool, aggregation_fn: Enum, num_mp_layers: int, num_out_layers, num_classes: int):
        super(mpnn, self).__init__()

        self.dim_proj = dim_proj
        self.num_nodes = num_nodes
        self.dropout_prob = dropout_prob
        self.skip_connections = skip_connections
        self.aggregation_function_fn = aggregation_fn

        self.mp_layer = [
            mp_layer(num_nodes, dim_proj, dropout_prob, skip_connections, aggregation_fn)
            for _ in range(num_mp_layers)
        ]

        self.out_layer = [
            nn.Linear(dim_proj, dim_proj)
            for _ in range(num_out_layers)
        ] + [nn.Linear(dim_proj, num_classes)]

    def __call__(self, connection_matrix, node_embeddings):
        assert node_embeddings.shape[1] == self.dim_proj, f'Incorrect node embedding size. Expected {self.dim_proj}, got {node_embeddings.shape[1]}'

        for mp_layer in self.mp_layer:
            node_embeddings = mp_layer(connection_matrix, node_embeddings)

        for out_layer in self.out_layer:
            node_embeddings = out_layer(node_embeddings)

        return node_embeddings

model_config = {
    'num_nodes': 2708,
    'dim_proj': 1433,
    'dropout_prob': 0.5,
    'skip_connections': True,
    'aggregation_fn': aggregation_fn.AVG,
    'num_mp_layers': 2,
    'num_out_layers': 1,
    'num_classes': 7
}

hyper_params = {
    'learning_rate': 3e-3,
    'num_epochs': 100
}

model = mpnn(**model_config)
mx.eval(model.parameters())


def pickle_read(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

data_dir = '../gat/data/CORA'

allx  = pickle_read(os.path.join(data_dir, 'allx'))
ally  = pickle_read(os.path.join(data_dir, 'ally'))
tx    = pickle_read(os.path.join(data_dir, 'tx'))
ty    = pickle_read(os.path.join(data_dir, 'ty'))

graph = pickle_read(os.path.join(data_dir, 'graph'))

test_idx = np.loadtxt(os.path.join(data_dir, 'test_index'), dtype=int)

allx = mx.array(allx.todense())
ally = mx.array(ally)
tx = mx.array(tx.todense())
ty = mx.array(ty)
test_idx = mx.array(test_idx)

node_embeddings = mx.concatenate([allx, tx], axis=0)

node_embeddings[2000].shape
ground_truth = mx.concat([ally, ty], axis=0)

def generate_connection_matrix(graph):
    edges = [(src, trg) for src, trgs in graph.items() for trg in trgs]
    src_idx, trg_idx = zip(*edges)
    return mx.array([src_idx, trg_idx])

connection_matrix = generate_connection_matrix(graph)
data = node_embeddings, connection_matrix

node_embeddings.shape[1]
model(node_embeddings, connection_matrix)
