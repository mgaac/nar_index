import mlx.core as mx
import mlx.nn as nn

import mlx.optimizers as optim
from mlx.utils import tree_map
from functools import partial

import numpy as np
import pickle
import os

class gat_layer(nn.Module):
    def __init__(self, num_nodes: int, dim_proj: int, num_att_heads: int, dropout_prob: float):
        super().__init__()

        self.source_dim = 0;
        self.target_dim = 1;

        self.dim_proj = dim_proj
        self.num_nodes = num_nodes
        self.num_att_heads = num_att_heads

        self.source_scores_fn = mx.random.normal([1, num_att_heads, dim_proj]) # Allegedly trianable parameters
        self.target_scores_fn = mx.random.normal([1, num_att_heads, dim_proj])

        self.leakyReLU = nn.LeakyReLU(0.02)
        self.dropout = nn.Dropout(dropout_prob)

    def __call__(self, node_proj, adjacency_matrix):

        source_idx = adjacency_matrix[self.source_dim]
        target_idx = adjacency_matrix[self.target_dim]

        node_proj = node_proj.reshape([-1, self.num_att_heads, self.dim_proj])

        source_scores = (node_proj * self.source_scores_fn).sum(axis=-1)
        target_scores = (node_proj * self.target_scores_fn).sum(axis=-1)

        edge_filtered_node_proj = mx.take(node_proj, source_idx, axis=0);

        edge_filtered_source_scores = mx.take(source_scores, source_idx, axis=0);
        edge_filtered_target_scores = mx.take(target_scores, target_idx, axis=0);

        edge_scores = self.leakyReLU(edge_filtered_source_scores + edge_filtered_target_scores)
        edge_scores = (edge_scores - edge_scores.max()).exp()

        softmax_denominator = mx.zeros([self.num_nodes, self.num_att_heads])

        softmax_denominator = softmax_denominator.at[target_idx].add(edge_scores)
        softmax_denominator = mx.take(softmax_denominator, target_idx, axis=0)

        attention_scores = edge_scores / (softmax_denominator + 1e-16)
        attention_scores = self.dropout(attention_scores)

        edge_filtered_node_proj = edge_filtered_node_proj * mx.expand_dims(attention_scores, axis=-1);

        new_node_proj = mx.zeros([self.num_nodes, self.num_att_heads, self.dim_proj]).at[target_idx].add(edge_filtered_node_proj)
        new_node_proj = self.leakyReLU(new_node_proj)

        return new_node_proj.reshape((self.num_nodes, self.num_att_heads * self.dim_proj))


class gat(nn.Module):
    def __init__(self, num_nodes: int, dim_embed: int, dim_proj: int, num_att_heads: int, num_layers: int, skip_connections: bool, dropout_prob: float,
                num_out_layers: int, num_out_classes: int):
        super().__init__()

        total_att_size = dim_proj * num_att_heads;

        self.dim_embed = dim_embed

        self.embed_proj = nn.Linear(dim_embed, total_att_size)

        self.gat_layers = [
          gat_layer(num_nodes, dim_proj, num_att_heads, dropout_prob)
          for _ in range(num_layers)
        ]

        self.out_layers = [
          nn.Linear(total_att_size, total_att_size)
          for _ in range(num_out_layers)
        ] + [nn.Linear(total_att_size, num_out_classes)]

        self.leakyReLU = nn.LeakyReLU(.02)
        self.dropout = nn.Dropout(dropout_prob)
        self.skip_connections = skip_connections

    def __call__(self, data):
        node_embeddings, adjacency_matrix = data

        assert node_embeddings.shape[1] == self.dim_embed, f'Incorrect node embedding size'

        node_proj = self.embed_proj(node_embeddings);
        node_proj = self.dropout(node_proj)

        for layer in self.gat_layers:
            new_node_proj = layer(node_proj, adjacency_matrix)
            if (self.skip_connections):
                new_node_proj += node_proj;
            node_proj = new_node_proj

        for layer in self.out_layers:
            node_proj = layer(node_proj)

        return node_proj


# example config and model instantiation
model_config = {
    'num_nodes': 2708,
    'dim_embed': 1433,
    'dim_proj': 8,
    'num_att_heads': 8,
    'num_layers': 2,
    'skip_connections': True,
    'dropout_prob': 0.5,
    'num_out_layers': 1,
    'num_out_classes': 7
}

hyper_params = {
    'learning_rate': 3e-3,
    'num_epochs': 100
}

gat_model = gat(**model_config)
mx.eval(gat_model.parameters())

def pickle_read(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

data_dir = 'data'

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

ground_truth = mx.concat([ally, ty], axis=0)

def generate_connection_matrix(graph):
    edges = [(src, trg) for src, trgs in graph.items() for trg in trgs]
    src_idx, trg_idx = zip(*edges)
    return mx.array([src_idx, trg_idx])

connection_matrix = generate_connection_matrix(graph)
data = node_embeddings, connection_matrix

def loss_fn(model, data, ground_truth):
    logits = model(data)
    return mx.mean(nn.losses.cross_entropy(logits, ground_truth, axis=1))

def eval_fn(model, data, ground_truth):
    logits = model(data)
    return mx.mean(mx.argmax(logits, axis=-1) == mx.argmax(ground_truth, axis=-1))

loss_and_grad_fn = nn.value_and_grad(gat_model, loss_fn)
optimizer = optim.Adam(learning_rate=hyper_params['learning_rate'])

for i in range(100000):
    loss, grads = loss_and_grad_fn(gat_model, data, ground_truth)
    optimizer.update(gat_model, grads)
    mx.eval(gat_model.parameters(), optimizer.state)

    if (i % 100 == 0):
        print(f'Epoch {i}: Loss {loss:.4f}, Accuracy {eval_fn(gat_model, data, ground_truth):.4f}')
