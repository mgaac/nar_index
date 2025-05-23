import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import networkx as nx
import numpy as np

from enum import Enum
from utils.datasets.nega_custom.scripts.graph_generator import load_graphs

class task(Enum):
    PARALLEL_ALGORITHM=0
    SEQUENTIAL_ALGORITHM=1
 
# Model Architecture
class aggregation_fn(Enum):
    SUM = 1
    AVG = 2
    MIN = 4
    MAX = 5

class mp_layer(nn.Module):
    def __init__(self, embedding_dim: int, dim_proj: int, dropout_prob: float, skip_connections: bool, aggregation_fn: Enum):
        super().__init__()

        self.source_idx = 0
        self.target_idx = 1

        self.dim_proj = dim_proj
        self.dropout_prob = dropout_prob
        self.skip_connections = skip_connections
        self.aggregation_fn = aggregation_fn

        self.source_message_fn = mx.random.normal([1, embedding_dim, dim_proj])
        self.target_message_fn = mx.random.normal([1, embedding_dim, dim_proj])

        self.update_fn = nn.Linear(dim_proj, embedding_dim)

        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def __call__(self, connection_matrix, node_embeddings):

        num_nodes = node_embeddings.shape[0]

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

        agg_message = mx.zeros([num_nodes, self.dim_proj])

        if (self.aggregation_fn == aggregation_fn.SUM):
            agg_message = agg_message.at[target_idx].add(message)

        elif (self.aggregation_fn == aggregation_fn.AVG):
            agg_message = agg_message.at[target_idx].add(message)
            denominator = mx.zeros([num_nodes, 1]).at[target_idx].add(1)
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
    def __init__(self, embedding_dim: int, dim_proj: int, dropout_prob: float, skip_connections: bool, aggregation_fn: Enum, num_mp_layers: int):
        super(mpnn, self).__init__()

        self.embedding_dim = embedding_dim
        self.dim_proj = dim_proj
        self.dropout_prob = dropout_prob
        self.skip_connections = skip_connections
        self.aggregation_function_fn = aggregation_fn

        self.mp_layer = [
            mp_layer(embedding_dim, dim_proj, dropout_prob, skip_connections, aggregation_fn)
            for _ in range(num_mp_layers)
        ]

    def __call__(self, data):
        node_embeddings, connection_matrix = data

        assert node_embeddings.shape[1] == self.embedding_dim, f'Incorrect node embedding size. Expected {self.embedding_dim}, got {node_embeddings.shape[1]}'

        for mp_layer in self.mp_layer:
            node_embeddings = mp_layer(connection_matrix, node_embeddings)

        return node_embeddings
    
class decoder(nn.Module):
    def __init__(self, embedding_dim: int, task_type: Enum):
        super(decoder, self).__init__()

        self.source_idx = 0
        self.target_idx = 1

        self.embedding_dim = embedding_dim

        self.predesecor_prob = nn.Linear(2 * embedding_dim, 1)

        if (task_type == task.PARALLEL_ALGORITHM):
            self.other_outputs = nn.Linear(embedding_dim, 2)
        elif (task_type == task.SEQUENTIAL_ALGORITHM):
            self.other_outputs = nn.Linear(embedding_dim, 1)

    def __call__(self, data):
        node_embeddings, connection_matrix = data

        num_nodes = node_embeddings.shape[0]

        edge_weights = mx.expand_dims(connection_matrix[2], axis=0)
        edge_weights = mx.expand_dims(edge_weights, axis=-1)

        connection_matrix = connection_matrix[:2].astype(mx.int32)

        source_idx = connection_matrix[self.source_idx]
        target_idx = connection_matrix[self.target_idx]

        source_embeddings = mx.take(node_embeddings, source_idx, axis=0)
        target_embeddings = mx.take(node_embeddings, target_idx, axis=0)

        concatenated_embeddings = mx.concat([source_embeddings, target_embeddings], axis=1)

        edge_scores = nn.relu(self.predesecor_prob(concatenated_embeddings))
        edge_scores = (edge_scores - edge_scores.max()).exp()

        softmax_denominator = mx.zeros([num_nodes, 1])

        softmax_denominator = softmax_denominator.at[target_idx].add(edge_scores)
        softmax_denominator = mx.take(softmax_denominator, target_idx, axis=0)

        edge_prob = edge_scores / (softmax_denominator + 1e-16)

        predesecor_predictions = mx.zeros([num_nodes, num_nodes])

        predesecor_predictions = predesecor_predictions.at[target_idx, source_idx].add(edge_prob.squeeze())

        other_predictions = self.other_outputs(node_embeddings)

        return other_predictions, predesecor_predictions
    
class nge(nn.Module):
    def __init__(self, embedding_dim: int, dim_proj: int, dropout_prob: float, skip_connections: bool, aggregation_fn: Enum, num_mp_layers: int):
        super(nge, self).__init__()

        self.parallel_encoder = nn.Linear(2, embedding_dim)
        self.sequential_encoder = nn.Linear(2, embedding_dim)

        self.parallel_decoder = decoder(embedding_dim, task.PARALLEL_ALGORITHM)
        self.sequential_decoder = decoder(embedding_dim, task.SEQUENTIAL_ALGORITHM)

        self.parallel_termination_node = nn.Linear(embedding_dim, 1, bias=False)
        self.parallel_termination_global = nn.Linear(embedding_dim, 1, bias=False)
        self.parallel_termination_bias = mx.zeros([1])

        self.sequential_termination_node = nn.Linear(embedding_dim, 1, bias=False)
        self.sequential_termination_global = nn.Linear(embedding_dim, 1, bias=False)
        self.sequential_termination_bias = mx.zeros([1])
    

        self.processor = mpnn(embedding_dim, dim_proj, dropout_prob, skip_connections, aggregation_fn, num_mp_layers)

    def __call__(self, data, task_type):
        node_embeddings, connection_matrix = data

        if task_type == task.PARALLEL_ALGORITHM:
            node_embeddings = self.parallel_encoder(node_embeddings)
            new_node_embeddings = self.processor((node_embeddings, connection_matrix))
            output = self.parallel_decoder((new_node_embeddings, connection_matrix))
        elif task_type == task.SEQUENTIAL_ALGORITHM:
            node_embeddings = self.sequential_encoder(node_embeddings)
            new_node_embeddings = self.processor((node_embeddings, connection_matrix))
            output = self.sequential_decoder((new_node_embeddings, connection_matrix))

        avg_node_embeddings = mx.mean(new_node_embeddings, axis=0)

        if task_type == task.PARALLEL_ALGORITHM:
            termination_prob = self.parallel_termination_node(new_node_embeddings) + self.parallel_termination_global(avg_node_embeddings) + self.parallel_termination_bias
            termination_prob = mx.mean(termination_prob)
            return output, termination_prob
        elif task_type == task.SEQUENTIAL_ALGORITHM:
            termination_prob = self.sequential_termination_node(new_node_embeddings) + self.sequential_termination_global(avg_node_embeddings) + self.sequential_termination_bias
            termination_prob = mx.mean(termination_prob)
            return output, termination_prob


#Dataset generation
train_graphs = load_graphs('train_graphs.pkl')
# val_graphs = load_graphs('val_graphs.pkl')
# test_graphs_20 = load_graphs('test_graphs_20.pkl')
# test_graphs_50 = load_graphs('test_graphs_50.pkl')
# test_graphs_100 = load_graphs('test_graphs_100.pkl')


model_config = {
    'embedding_dim': 100,
    'dim_proj': 10,
    'dropout_prob': 0.5,
    'skip_connections': True,
    'aggregation_fn': aggregation_fn.MAX,
    'num_mp_layers': 1
}

hyper_params = {
    'learning_rate': 0.001,
    'num_graphs': 10
}

model = nge(**model_config)
optimizer = optim.Adam(learning_rate=hyper_params["learning_rate"])

def sequential_loss_fn(model, input, graph_targets, termination_target):
    output, termination_prob = model(input, task.SEQUENTIAL_ALGORITHM)

    termination_prob = mx.expand_dims(termination_prob, axis=0)

    state, predesecor = output
    reachability_target, predesecor_target = graph_targets

    state_loss = nn.losses.binary_cross_entropy(state, mx.expand_dims(reachability_target, axis=1))

    pred_loss = nn.losses.cross_entropy(predesecor, predesecor_target, reduction='sum')
    termination_loss = nn.losses.binary_cross_entropy(termination_prob, termination_target)

    total_loss = state_loss + pred_loss + termination_loss

    return total_loss, output


loss_and_grad_fn = nn.value_and_grad(model, sequential_loss_fn)

def train_step(model, input, graph_targets, termination_target, debug=False):
    (loss, output), grads = loss_and_grad_fn(
        model, input, graph_targets, termination_target
    )
    optimizer.update(model, grads)
    
    if debug:
        # Compute individual loss components for debugging (using exact same logic as loss function)
        output_debug, termination_prob_debug = model(input, task.SEQUENTIAL_ALGORITHM)
        termination_prob_debug = mx.expand_dims(termination_prob_debug, axis=0)
        
        state, predecessor = output_debug
        state_target, predecessor_target = graph_targets
        
        # Compute losses exactly as in sequential_loss_fn
        state_loss = nn.losses.binary_cross_entropy(state, mx.expand_dims(state_target, axis=1))
        pred_loss = nn.losses.cross_entropy(predecessor, predecessor_target, reduction='sum')  # Same as loss function
        termination_loss = nn.losses.binary_cross_entropy(termination_prob_debug, termination_target)
        
        debug_info = {
            'targets': {'state': state_target, 'predecessor': predecessor_target, 'termination': termination_target},
            'outputs': {'state': state, 'predecessor': predecessor, 'termination': termination_prob_debug},
            'losses': {'state': float(state_loss.item()), 'predecessor': float(pred_loss.item()), 'termination': float(termination_loss.item())}
        }
        return loss, output, debug_info
    
    return loss, output


def print_debug_info(graph_idx, step_idx, targets, outputs, losses, total_loss):
    """Print comprehensively formatted debugging information for model training analysis"""
    print("=" * 80)
    print(f"TRAINING STEP DEBUG ANALYSIS - Graph {graph_idx:02d} | Step {step_idx:02d}")
    print("=" * 80)
    
    # Print targets section
    print("GROUND TRUTH TARGETS:")
    print("-" * 50)
    print(f"  State Target (Reachability):     {targets['state']}")
    print(f"  Predecessor Target (Node IDs):   {targets['predecessor']}")
    print(f"  Termination Target (Binary):     {targets['termination']}")
    print()
    
    # Print model outputs section
    print("MODEL PREDICTIONS:")
    print("-" * 50)
    print(f"  State Prediction (Logits):       {outputs['state']}")
    print(f"  Predecessor Prediction (Logits): {outputs['predecessor']}")
    print(f"  Termination Prediction (Prob):   {outputs['termination']}")
    print()
    
    # Print losses section
    print("LOSS COMPONENT ANALYSIS:")
    print("-" * 50)
    print(f"  Binary Cross-Entropy (State):    {losses['state']:.6f}")
    print(f"  Cross-Entropy (Predecessor):     {losses['predecessor']:.6f}")
    print(f"  Binary Cross-Entropy (Term):     {losses['termination']:.6f}")
    print("-" * 50)
    print(f"  COMBINED TOTAL LOSS:             {total_loss:.6f}")
    total_check = losses['state'] + losses['predecessor'] + losses['termination']
    print(f"  VERIFICATION (Sum of parts):     {total_check:.6f}")
    print("=" * 80)
    print()


def train_sequential_model(model, dataset, num_graphs, debug=False):
    print("INITIATING SEQUENTIAL MODEL TRAINING PROTOCOL")
    print("=" * 80)
    
    for graph_idx in range(num_graphs):
        print(f"PROCESSING GRAPH {graph_idx + 1} OF {num_graphs}")
        print(f"Graph Identifier: {graph_idx}")
        
        graph_execution_history = dataset[graph_idx]['targets']['sequential']
        connection_matrix = dataset[graph_idx]['connection_matrix']
        
        residual_features = mx.zeros([len(graph_execution_history['prim_state'][0])])
        num_steps = len(graph_execution_history['prim_state']) - 1
        
        print(f"Total training steps for this graph: {num_steps}")
        print(f"Node count: {len(graph_execution_history['prim_state'][0])}")
        print()
        
        for i in range(num_steps):
            # Prepare targets
            prim_state_target = graph_execution_history['prim_state'][i + 1]
            prim_predecessor_target = graph_execution_history['prim_predecessor'][i + 1]
            termination_target = mx.ones([1]) if i == num_steps - 1 else mx.zeros([1])
            
            # Prepare input
            current_features = graph_execution_history['prim_state'][i]
            input_features = mx.stack([current_features, residual_features], axis=1)
            input_data = (input_features, connection_matrix)
            
            graph_targets = (prim_state_target, prim_predecessor_target)
            
            # Train step with optional debugging
            if debug:
                loss, output, debug_info = train_step(
                    model, input_data, graph_targets, termination_target, debug=True
                )
                print_debug_info(
                    graph_idx + 1, i + 1, 
                    debug_info['targets'], debug_info['outputs'], debug_info['losses'], 
                    float(loss)
                )
            else:
                loss, output = train_step(
                    model, input_data, graph_targets, termination_target, debug=False
                )
            
            # Update residual features for next step
            state, predecessor = output
            residual_features = state[:, 0]
        
        print(f"GRAPH {graph_idx + 1} TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()

train_sequential_model(model, train_graphs, 1, debug=True)




            


            






