import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from model import mpnn, aggregation_fn
from utils.datasets.CORA.scripts.data_loading import load_data

from functools import partial

from tqdm import tqdm

def loss_fn(model, data, labels, mask):
    model = model.train()
    logits = model(data)
    loss = nn.losses.cross_entropy(logits, labels, axis=1) * mask
    n_samples = mask.sum().sum()
    return loss.sum() / n_samples

def eval_fn(model, data, labels):
    model = model.eval()
    logits = model(data)
    logits = mx.softmax(logits, axis=-1)
    return mx.mean(mx.argmax(logits, axis=-1) == mx.argmax(labels, axis=-1))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GAT model on the CORA dataset"
    )
    parser.add_argument(
        "--aggregation_fn", type=aggregation_fn, default=aggregation_fn.MAX,
        help="Total number of training steps"
    )
    parser.add_argument(
        "--data", type=str, default="../../utils/datasets/CORA/data",
        help="Path to the dataset folder"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-3,
        help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--num-steps", type=int, default=10000,
        help="Total number of training steps"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Model configuration
    model_config = {
        'num_nodes': 2708,
        'embedding_dim': 1433,
        'dim_proj': 8,
        'dropout_prob': 0.5,
        'skip_connections': True,
        'aggregation_fn': args.aggregation_fn,
        'num_mp_layers': 1,
        'num_out_layers': 1,
        'num_classes': 7
    }

    # Hyperparameters from CLI
    hyper_params = {
        'learning_rate': args.learning_rate,
        'num_steps': args.num_steps
    }

    # Initialize model and optimizer
    model = mpnn(**model_config)
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=hyper_params['learning_rate'])

    # Prepare loss/gradient function
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Load data
    node_embeddings, connection_matrix, labels, train_mask, test_mask = load_data(args.data)
    data = (node_embeddings, connection_matrix)

    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(node_embeddings, connection_matrix, labels, mask):
        loss, grads = loss_and_grad_fn(model, (node_embeddings, connection_matrix), labels, mask)
        optimizer.update(model, grads)

        return loss

    # Training loop
    pbar = tqdm(range(hyper_params['num_steps']), desc="Training", unit="iters")
    for i in pbar:
        loss = step(node_embeddings, connection_matrix, labels, train_mask)

        # Periodic evaluation
        if i % 50 == 0:
            test_loss = loss_fn(model, data, labels, test_mask)
            accuracy = eval_fn(model, data, labels)

        # Update tqdm status
        if i % 50 == 0:
            pbar.set_postfix(
                train_loss=f"{loss:.4f}",
                test_loss=f"{test_loss:.4f}",
                accuracy=f"{accuracy:.4f}"
            )

if __name__ == "__main__":
    main()
