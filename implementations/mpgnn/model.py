import mlx.core as mx
import mlx.nn as nn

from enum import Enum

class aggregation_fn(Enum):
    SUM = 1
    AVG = 2
    MAX = 3
    MIN = 4

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

        connection_matrix = self.dropout(connection_matrix)
        node_embeddings = self.dropout(node_embeddings)

        print(connection_matrix)

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
