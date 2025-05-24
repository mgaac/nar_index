import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import networkx as nx
import numpy as np
from tqdm import tqdm

from enum import Enum
from utils.datasets.nega_custom.scripts.graph_generator import load_graphs

# Training Logger Classes
class TrainingLogger:
    """Base class for training loggers"""
    def __init__(self, debug=False):
        self.debug = debug
    
    def start_training(self, num_graphs):
        """Called at the start of training"""
        pass
    
    def start_graph(self, graph_idx, num_graphs, num_steps):
        """Called at the start of each graph"""
        pass
    
    def start_step(self, step_idx, num_steps):
        """Called at the start of each step"""
        pass
    
    def log_losses(self, loss, state_loss, pred_loss, termination_loss):
        """Called to log losses for current step"""
        pass
    
    def log_debug_info(self, state, predesecor, reachability_target, predesecor_target, termination_prob, termination_target):
        """Called to log debug information"""
        if self.debug:
            self._print_debug_info(state, predesecor, reachability_target, predesecor_target, termination_prob, termination_target)
    
    def end_graph(self, graph_idx, avg_loss):
        """Called at the end of each graph"""
        pass
    
    def end_training(self, overall_avg_loss):
        """Called at the end of training"""
        pass
    
    def _print_debug_info(self, state, predesecor, reachability_target, predesecor_target, termination_prob, termination_target):
        """Print detailed debug information comparing predictions with targets"""
        print("\n    DEBUG INFO:")
        print("    " + "-" * 40)
        print(f"    Predecessor Predictions: {np.array(mx.argmax(predesecor, axis=1))}")
        print(f"    Predecessor Targets:     {np.array(predesecor_target)}")
        print(f"    State Predictions:       {np.array(mx.argmax(state, axis=1))}")
        print(f"    State Targets:           {np.array(mx.argmax(reachability_target, axis=1))}")
        print(f"    Termination Predictions: {np.array(termination_prob)}")
        print(f"    Termination Probability: {float(mx.softmax(termination_prob, axis=0)[1]):.4f}")
        print(f"    Termination Targets:     {np.array(termination_target)}")
        print("    " + "-" * 40)

class PrintLogger(TrainingLogger):
    """Logger that uses organized print statements"""
    def start_training(self, num_graphs):
        print("=" * 60)
        print("STARTING SEQUENTIAL MODEL TRAINING")
        print(f"Number of graphs: {num_graphs}")
        print(f"Debug mode: {'ON' if self.debug else 'OFF'}")
        print("=" * 60)
    
    def start_graph(self, graph_idx, num_graphs, num_steps):
        print(f"\n--- Training on Graph {graph_idx + 1}/{num_graphs} ---")
        print(f"Graph has {num_steps} training steps")
    
    def start_step(self, step_idx, num_steps):
        print(f"\n  Step {step_idx + 1}/{num_steps}:")
    
    def log_losses(self, loss, state_loss, pred_loss, termination_loss):
        print(f"    Losses:")
        print(f"      State Loss:       {state_loss:.6f}")
        print(f"      Predecessor Loss: {pred_loss:.6f}")
        print(f"      Termination Loss: {termination_loss:.6f}")
        print(f"      Total Loss:       {loss:.6f}")
    
    def end_graph(self, graph_idx, avg_loss, num_steps=None):
        print(f"\n  Graph {graph_idx + 1} Average Loss: {avg_loss:.6f}")
    
    def end_training(self, overall_avg_loss):
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print(f"Overall Average Loss: {overall_avg_loss:.6f}")
        print("=" * 60)

class TqdmLogger(TrainingLogger):
    """Logger that uses tqdm progress bars"""
    def __init__(self, debug=False):
        super().__init__(debug)
        self.graph_iterator = None
        self.step_iterator = None
    
    def start_training(self, num_graphs):
        self.graph_iterator = tqdm(range(num_graphs), desc="Training Graphs", unit="graph")
    
    def start_graph(self, graph_idx, num_graphs, num_steps):
        self.step_iterator = tqdm(range(num_steps), desc=f"Graph {graph_idx + 1} Steps", unit="step", leave=False)
    
    def start_step(self, step_idx, num_steps):
        pass  # tqdm handles step display automatically
    
    def log_losses(self, loss, state_loss, pred_loss, termination_loss):
        if self.step_iterator:
            self.step_iterator.set_postfix({
                'Loss': f'{float(loss):.4f}',
                'State': f'{float(state_loss):.4f}',
                'Pred': f'{float(pred_loss):.4f}',
                'Term': f'{float(termination_loss):.4f}'
            })
            self.step_iterator.update(1)
    
    def end_graph(self, graph_idx, avg_loss, num_steps=None):
        if self.step_iterator:
            self.step_iterator.close()
        if self.graph_iterator:
            self.graph_iterator.set_postfix({
                'Avg Loss': f'{avg_loss:.4f}',
                'Steps': num_steps or 0
            })
            self.graph_iterator.update(1)
    
    def end_training(self, overall_avg_loss):
        if self.graph_iterator:
            self.graph_iterator.close()
        print(f"\nTraining completed! Overall Average Loss: {overall_avg_loss:.6f}")

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
            self.state_outputs = nn.Linear(embedding_dim, 2)
        elif (task_type == task.SEQUENTIAL_ALGORITHM):
            self.state_outputs = nn.Linear(embedding_dim, 2)

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

        state_predictions = self.state_outputs(node_embeddings)

        return state_predictions, predesecor_predictions
    
class nge(nn.Module):
    def __init__(self, embedding_dim: int, dim_proj: int, dropout_prob: float, skip_connections: bool, aggregation_fn: Enum, num_mp_layers: int):
        super(nge, self).__init__()

        self.parallel_encoder = nn.Linear(2, embedding_dim)
        self.sequential_encoder = nn.Linear(2, embedding_dim)

        self.parallel_decoder = decoder(embedding_dim, task.PARALLEL_ALGORITHM)
        self.sequential_decoder = decoder(embedding_dim, task.SEQUENTIAL_ALGORITHM)

        self.parallel_termination_node = nn.Linear(embedding_dim, 2, bias=False)
        self.parallel_termination_global = nn.Linear(embedding_dim, 2, bias=False)
        self.parallel_termination_bias = mx.random.normal([2])

        self.sequential_termination_node = nn.Linear(embedding_dim, 2, bias=False)
        self.sequential_termination_global = nn.Linear(embedding_dim, 2, bias=False)
        self.sequential_termination_bias = mx.random.normal([2])
    

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
            termination_prob = mx.mean(termination_prob, axis=0)
            return output, termination_prob
        elif task_type == task.SEQUENTIAL_ALGORITHM:
            termination_prob = self.sequential_termination_node(new_node_embeddings) + self.sequential_termination_global(avg_node_embeddings) + self.sequential_termination_bias
            termination_prob = mx.mean(termination_prob, axis=0)
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

def sequential_loss_fn(model, input, graph_targets, termination_target, logger=None):
    output, termination_prob = model(input, task.SEQUENTIAL_ALGORITHM)

    state, predesecor = output
    reachability_target, predesecor_target = graph_targets

    state_loss = nn.losses.binary_cross_entropy(state, reachability_target, reduction='mean')

    pred_loss = nn.losses.cross_entropy(predesecor, predesecor_target, reduction='mean')
    termination_loss = nn.losses.binary_cross_entropy(termination_prob, termination_target, reduction='mean')

    total_loss = state_loss + pred_loss + termination_loss

    if logger:
        logger.log_debug_info(state, predesecor, reachability_target, predesecor_target, termination_prob, termination_target)

    return total_loss, (state_loss, pred_loss, termination_loss), output, termination_prob


loss_and_grad_fn = nn.value_and_grad(model, sequential_loss_fn)

def train_step(model, input, graph_targets, termination_target, logger=None):
    (loss, (state_loss, pred_loss, termination_loss), output, termination_prob), grads = loss_and_grad_fn(
        model, input, graph_targets, termination_target, logger
    )
    optimizer.update(model, grads)
    
    return loss, (state_loss, pred_loss, termination_loss), output, termination_prob


def train_sequential_model(model, dataset, num_graphs, logger):
    """Clean training function that focuses only on training logic"""
    logger.start_training(num_graphs)
    
    total_graphs_loss = 0.0
    valid_graphs = 0
    
    for graph_idx in range(num_graphs):
        graph_execution_history = dataset[graph_idx]['targets']['sequential']
        connection_matrix = dataset[graph_idx]['connection_matrix']
        
        residual_features = mx.zeros([len(graph_execution_history['prim_state'][0])])
        num_steps = len(graph_execution_history['prim_state']) - 1

        if num_steps == 0:
            continue
        
        valid_graphs += 1
        logger.start_graph(graph_idx, num_graphs, num_steps)
        
        graph_total_loss = 0.0
        
        for i in range(num_steps):
            logger.start_step(i, num_steps)
            
            # Prepare training data
            prim_state_target = graph_execution_history['prim_state'][i + 1]
            prim_predecessor_target = graph_execution_history['prim_predecessor'][i + 1]
            termination_target = graph_execution_history['prim_termination'][i + 1]
            
            current_features = mx.argmax(graph_execution_history['prim_state'][i], axis=1)
            input_features = mx.stack([current_features, residual_features], axis=1)
            input_data = (input_features, connection_matrix)
            graph_targets = (prim_state_target, prim_predecessor_target)
            
            # Training step
            (loss, (state_loss, pred_loss, termination_loss), output, termination_prob) = train_step(
                model, input_data, graph_targets, termination_target, logger
            )

            # Update residual features for next step
            state, _ = output


            if mx.softmax(termination_prob, axis=0)[1] > 0.5:
                break

            residual_features = mx.argmax(state, axis=1)

            # Log losses
            graph_total_loss += float(loss)
            logger.log_losses(loss, state_loss, pred_loss, termination_loss)
        
        # End of graph
        avg_graph_loss = graph_total_loss / num_steps
        total_graphs_loss += avg_graph_loss
        logger.end_graph(graph_idx, avg_graph_loss, num_steps)
    
    # End of training
    overall_avg_loss = total_graphs_loss / valid_graphs if valid_graphs > 0 else 0.0
    logger.end_training(overall_avg_loss)


print_logger = TqdmLogger(debug=False)  
train_sequential_model(model, train_graphs, 700, print_logger)

            


            






