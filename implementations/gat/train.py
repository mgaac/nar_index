import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from model import gat
from data_loading import load_data

from functools import partial
from tqdm import tqdm

def loss_fn(model, data, ground_truth):
    logits = model(data)
    return mx.mean(nn.losses.cross_entropy(logits, ground_truth, axis=1))

def eval_fn(model, data, ground_truth):
    logits = model(data)
    return mx.mean(mx.argmax(logits, axis=-1) == mx.argmax(ground_truth, axis=-1))

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
    'num_steps': 100
}

model = gat(**model_config)

optimizer = optim.Adam(learning_rate=hyper_params['learning_rate'])
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

@partial(mx.compile, inputs=model.state, outputs=[model.state, optimizer.state])
def step(data, labels):
    loss, grads = loss_and_grad_fn(model, data, labels)
    optimizer.update(model, grads)

    return loss

dataset = load_data('data/CORA')

node_embeddings, connection_matrix, ground_truth, train_mask, test_mask = dataset
data = node_embeddings, connection_matrix

train_labels = ground_truth * train_mask
test_labels = ground_truth * test_mask



pbar = tqdm(range(hyper_params['num_steps']), desc="Training", unit="iters")
for i in pbar:
    loss = step(data, train_labels)
    mx.eval(model.state, optimizer.state)

    eval = 0.0
    if i % 100 == 0:
        eval = eval_fn(model, data, test_labels)

    pbar.set_postfix(loss=f"{loss:.4f}", eval=f"{eval:.4f}")
