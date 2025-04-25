import mlx.core as mx
import mlx.nn as nn

import mlx.optimizers as optim
from mlx.utils import tree_map
from functools import partial

from enum import Enum

import numpy as np
import pickle
import os

class aggregation_function(Enum):
    SUM = 1
    AVG = 2
    MAX = 3
    MIN = 4


class message_function(nn.Module):
    def __init__(self, num_nodes: int, dim_proj: int, dropout_prob: float, skip_connections: bool):





class mpnn(nn.Module):
    def __init__(self, num_nodes: int, dim_proj: int, dropout_prob: float, skip_connections: bool, aggregation_function: Enum):
        super().__init__()

        self.num_nodes = num_nodes
        self.dim_proj = dim_proj
        self.dropout_prob = dropout_prob
        self.skip_connections = skip_connections
        self.aggregation_function = aggregation_function
