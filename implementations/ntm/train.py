from model import controller
import utils

import mlx.optimizers as optim
from mlx.utils import tree_map
from functools import partial
import mlx.core as mx
import mlx.nn as nn

from tqdm import tqdm
import argparse
import random
import csv
import json

network_param = {
    "idim": 5,
    "odim": 5,
    "hdim": 64,
    "numl_shared": 10,
    "numl_con": 10,
    "numl_out": 10,
    "memory_size": [64, 10]
}

train_param = {
    "l_rate": 6e-5
}

def init_fn(x, initializer=nn.init.he_normal()):
    if hasattr(x, 'shape') and len(x.shape) >= 2:
        return initializer(x)
    else:
        return x

def is_leaf(node):
    return isinstance(node, mx.array)

model = controller(**network_param)
new_params = tree_map(init_fn, model.state, is_leaf=is_leaf)
model.update(new_params)
optimizer = optim.Adam(train_param["l_rate"])

def loss_fn(logits, targets):
    return nn.losses.binary_cross_entropy(logits, targets)

def train_pair_iter(max_sequence_length, element_length):
    while True:
        copy_len = random.randint(1, max_sequence_length - 1)
        target = mx.random.randint(0, 2, shape=[copy_len, element_length])
        input = mx.concatenate([target, mx.ones([1, element_length])])
        input = mx.concatenate([input, mx.zeros([copy_len, element_length])])
        target = mx.concatenate([mx.zeros([copy_len + 1, element_length]), target])
        yield input, target, copy_len

def acc_fn(logits, target):
    logits = mx.sigmoid(logits)
    logits = mx.where(logits > .5, mx.ones_like(logits), mx.zeros_like(logits))
    acc = logits == target
    acc = acc.sum() / acc.size

    return acc

def copy_task_eval_fn(input, targets, copy_len):
    acc = []
    r = mx.zeros(network_param["memory_size"][1])
    w = mx.zeros(network_param["memory_size"][0])
    memory = mx.ones(network_param["memory_size"]) * 1e-6
    for i, sequence in enumerate(input):
        logits, r, w, memory, _ = model(sequence, r, w, memory)
        if (i > copy_len):
            acc.append(acc_fn(logits, targets[i]))
    return mx.array(acc).sum() / len(acc)

def forward_loss(input, target, r, w, memory, copy_len):
    loss_total = 0
    activations = []
    logits = None
    # Process each sequence (for simplicity, only the last logits will be used for logging)
    for i, sequence in enumerate(input):
        logits, r, w, memory, activations = model(sequence, r, w, memory)
        if (i > copy_len):
            loss_total += loss_fn(logits, target[i])
    loss = loss_total / len(input)
    return loss, r, w, memory, logits, activations


state = [model.state, optimizer.state]
@partial(mx.compile, inputs=state, outputs=state)
def step(input, target, r, w, memory, copy_len):
    value_and_grad_fn = nn.value_and_grad(model, forward_loss)
    (loss, r, w, memory, logits, activations), grads = value_and_grad_fn(input, target, r, w, memory, copy_len)
    grads, _ = optim.clip_grad_norm(grads, max_norm=2.0)
    optimizer.update(model, grads)
    return (loss, r, w, memory, logits, activations), grads

def train(iters, max_sequence_length, element_length, csv_file="run_log.csv"):
    # Make a copy of the initial parameters for update ratio computation.
    prev_params = {k: v.copy() for k, v in model.parameters().items()}
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = [
            "Iteration", "Training Loss", "Accuracy Score", "Learning Rate",
            "Gradient Stats", "Activation Stats", "Parameter Stats",
            "Weight Update Ratio", "Prediction vs Target"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        r = mx.ones(network_param["memory_size"][1]) * 1e-2
        w = mx.ones(network_param["memory_size"][0]) * 1e-2
        memory = mx.ones(network_param["memory_size"]) * 1e-6

        pbar = tqdm(range(iters), desc="Progress", unit=" iters")
        for i in pbar:
            input, target, copy_len = next(train_pair_iter(max_sequence_length, element_length))
            (loss, r, w, memory, logits, activations), grads = step(input, target, r, w, memory, copy_len)
            mx.eval(model.parameters(), optimizer.state)
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

            if i % 1000 == 0:
                # Compute accuracy along with predictions and targets for visualization.
                acc_value = copy_task_eval_fn(input, target, copy_len)
                # Compute additional statistics using the helper functions.
                grad_stats = utils.compute_gradient_stats(grads)
                act_stats = utils.compute_activation_stats(activations)
                param_stats = utils.compute_param_stats(model.parameters())
                update_ratios = utils.compute_update_ratios(prev_params, model.parameters())
                # Update the previous parameters.
                prev_params = {k: v.copy() for k, v in model.parameters().items()}
                learning_rate = optimizer.state.get("lr", train_param["l_rate"])

                log_data = {
                    "Iteration": i,
                    "Training Loss": loss.item(),
                    "Accuracy Score": acc_value.item(),
                    "Learning Rate": learning_rate,
                    "Gradient Stats": json.dumps(grad_stats),
                    "Activation Stats": json.dumps(act_stats),
                    "Parameter Stats": json.dumps(param_stats),
                    "Weight Update Ratio": json.dumps(update_ratios),
                }

                writer.writerow(log_data)
                csvfile.flush()

train(150000, 10, 5, csv_file="run_log/final_2.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=150000)
    parser.add_argument('--max_seq_len', type=int, default=10)
    parser.add_argument('--element_length', type=int, default=5)
    parser.add_argument('--csv_file', type=str, default="run_log/final_2.csv")
    args = parser.parse_args()
    train(args.iters, args.max_seq_len, args.element_length, csv_file=args.csv_file)
