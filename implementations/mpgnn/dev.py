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


        node_embeddings = self.dropout(node_embeddings)

        print(mx.maximum(connection_matrix, 2708))

        source_idx = connection_matrix[self.source_idx].astype(mx.int32)
        target_idx = connection_matrix[self.target_idx].astype(mx.int32)

        source_embeddings = node_embeddings @ self.source_message_fn
        target_embeddings = node_embeddings @ self.target_message_fn

        filtered_source_embeddings = mx.take(source_embeddings, source_idx)
        filtered_target_embeddings = mx.take(target_embeddings, target_idx)

        message = filtered_source_embeddings + filtered_target_embeddings
        message = self.relu(message)

        agg_message = mx.zeros([self.num_nodes, self.dim_proj])

        if (self.aggregation_fn == aggregation_fn.SUM):
            agg_message = agg_message.at[target_idx].add(message)

        elif (self.aggregation_fn == aggregation_fn.AVG):
            agg_message = agg_message.at[target_idx].add(message)
            denominator = mx.zeros([self.num_nodes, 1]).at[target_idx].add(1)

            print(agg_message.shape, denominator.shape, target_idx.shape)

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


model_config = {
    'num_nodes': 2708,
    'embedding_dim': 1433,
    'dim_proj': 8,
    'dropout_prob': 0.5,
    'skip_connections': True,
    'aggregation_fn': aggregation_fn.MAX,
    'num_mp_layers': 2,
    'num_out_layers': 0,
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

data_dir = '../datasets/CORA/data'

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
    return mx.array([src_idx, trg_idx], dtype=mx.int32)

connection_matrix = generate_connection_matrix(graph)
data = node_embeddings, connection_matrix

def loss_fn(model, data, ground_truth):
    logits = model(data)
    return mx.mean(nn.losses.cross_entropy(logits, ground_truth, axis=1))

def eval_fn(model, data, ground_truth):
    logits = model(data)
    return mx.mean(mx.argmax(logits, axis=-1) == mx.argmax(ground_truth, axis=-1))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.Adam(learning_rate=hyper_params['learning_rate'])

for i in range(100000):
    loss, grads = loss_and_grad_fn(model, data, ground_truth)
    optimizer.update(model, grads)
    mx.eval(model.state, optimizer.state)
    break

    if (i % 10 == 0):
        print(f'Epoch {i}: Loss {loss:.4f}, Accuracy {eval_fn(model, data, ground_truth):.4f}')


grads['out_layer'][0]
