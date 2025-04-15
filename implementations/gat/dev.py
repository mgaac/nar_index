import mlx.core as mx
import mlx.nn as nn

import mlx.optimizers as optim
from mlx.utils import tree_map
from functools import partial
from tqdm import tqdm
import numpy as np
import argparse
import random
import csv
import json


class gat_layer(nn.Module):
    def __init__(self, num_nodes: int, dim_proj: int, num_att_heads: int, dropout_prob: float):
        super().__init__()
        # indices into adjacency: 0 = source, 1 = target
        self.source_dim = 0
        self.target_dim = 1

        self.dim_proj = dim_proj
        self.num_nodes = num_nodes
        self.num_att_heads = num_att_heads

        # learnable projection vectors for attention (shape: [1, heads, proj_dim])
        self.source_scores_fn = mx.array([1, num_att_heads, dim_proj])
        self.target_scores_fn = mx.array([1, num_att_heads, dim_proj])

        # activation and dropout for attention weights
        self.leakyReLU = nn.LeakyReLU(0.02)
        self.dropout = nn.Dropout(dropout_prob)

    def __call__(self, node_proj, adjacency_matrix):
        # adjacency_matrix: [2, num_edges] containing source and target node indices
        source_idx = adjacency_matrix[self.source_dim]
        target_idx = adjacency_matrix[self.target_dim]

        # reshape node projections to [num_nodes, heads, proj_dim]
        node_proj = node_proj.reshape([-1, self.num_att_heads, self.dim_proj])

        # compute attention score components for each node
        source_scores = (node_proj * self.source_scores_fn).sum(dim=-1)  # [num_nodes, heads]
        target_scores = (node_proj * self.target_scores_fn).sum(dim=-1)

        # gather source and target for each edge
        edge_node_proj = mx.take(node_proj, source_idx, axis=0)           # [num_edges, heads, proj_dim]
        edge_src_scores = mx.take(source_scores, source_idx, axis=0)      # [num_edges, heads]
        edge_tgt_scores = mx.take(target_scores, target_idx, axis=0)

        # raw attention logits per edge
        edge_scores = self.leakyReLU(edge_src_scores + edge_tgt_scores)   # [num_edges, heads]
        edge_scores = (edge_scores - edge_scores.max()).exp()             # stabilize and exponentiate

        # compute normalization denominator per target node and head
        denom = mx.zeros([self.num_nodes, self.num_att_heads])
        denom = denom.at[target_idx].add(edge_scores)

        # normalized attention weights
        attn = edge_scores / (denom[target_idx] + 1e-16)
        attn = self.dropout(attn)

        # weight edge node projections by attention
        weighted_proj = edge_node_proj * attn[..., None]                  # [num_edges, heads, proj_dim]

        # aggregate back to target nodes
        new_proj = mx.zeros([self.num_nodes, self.num_att_heads, self.dim_proj])
        new_proj = new_proj.at[target_idx].add(weighted_proj)

        # apply activation and flatten back to [num_nodes, heads * proj_dim]
        new_proj = self.leakyReLU(new_proj).reshape((self.num_nodes, -1))

        return new_proj


class gat(nn.Module):
    def __init__(self, num_nodes: int, dim_embed: int, dim_proj: int, num_att_heads: int,
                 num_layers: int, skip_connections: bool, dropout_prob: float,
                 num_out_layers: int, num_out_classes: int):
        super().__init__()

        total_att_size = dim_proj * num_att_heads

        # initial linear projection from embedding to attention input
        self.embed_proj = nn.Linear(dim_embed, total_att_size)
        # stack of GAT layers
        self.gat_layers = nn.Sequential([gat_layer(num_nodes, dim_proj, num_att_heads, dropout_prob)
                                         for _ in range(num_layers)])
        # output MLP: some hidden layers then final classifier
        self.out_layers = nn.Sequential(
            *[nn.Linear(total_att_size, total_att_size) for _ in range(num_out_layers)],
            nn.Linear(total_att_size, num_out_classes)
        )

        self.leakyReLU = nn.LeakyReLU(.02)
        self.dropout = nn.Dropout(dropout_prob)
        self.skip_connections = skip_connections
        # save for dimension check
        self.dim_embed = dim_embed

    def __call__(self, node_embeddings, adjacency_matrix):
        # ensure correct input embedding size
        assert node_embeddings.shape[1] == self.dim_embed, \
            f'Expected embedding dim {self.dim_embed}, got {node_embeddings.shape[1]}'

        # project embeddings for attention
        node_proj = self.embed_proj(node_embeddings)
        node_proj = self.dropout(node_proj)

        # apply each GAT layer (with optional residual)
        for layer in self.gat_layers:
            updated = layer(node_proj, adjacency_matrix)
            if self.skip_connections:
                updated = updated + node_proj
            node_proj = updated

        # pass through final MLP
        out = node_proj
        for layer in self.out_layers:
            out = layer(out)

        # return class probabilities per node
        return mx.softmax(out)


# example config and model instantiation
model_config = {
    'num_nodes': 100,
    'dim_embed': 128,
    'dim_proj': 8,
    'num_att_heads': 8,
    'num_layers': 5,
    'skip_connections': True,
    'dropout_prob': 0.5,
    'num_out_layers': 1,
    'num_out_classes': 7
}

gat_model = gat(**model_config)
