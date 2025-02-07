from model import controller
from utils import visualize_training_history

import mlx.optimizers as optim
import mlx.core as mx
import mlx.nn as nn

from functools import partial

import matplotlib.pyplot as plt
import numpy as np

network_param = {
    "idim": 5,
    "odim": 5,
    "hdim": 64,
    "numl_shared": 3,
    "numl_con": 3,
    "numl_out": 3,
    "memory_size": [64, 10]
}

train_param = {
    "l_rate": 1e-3
}

def loss_fn(logits, targets):
    return nn.losses.binary_cross_entropy(logits, targets)

def eval_fn(logits, targets):
    return mx.sum(1 - (logits - targets)) / logits.shape[0]

def train_pair_iter(max_sequence_length, element_length):
    while(True):
        copy_len = mx.random.randint(1, max_sequence_length)
        target = mx.random.randint(0, 2, shape=[copy_len, element_length])
        input = mx.concatenate([target, mx.ones([1, element_length])])
        input = mx.concatenate([input, mx.zeros([copy_len, element_length])])
        target = mx.concatenate([mx.zeros([copy_len + 1, element_length]), target])
        yield input, target, copy_len.item()

model = controller(**network_param)
optimizer = optim.Adam(learning_rate=train_param["l_rate"])
state = [model.state, optimizer.state]

init_fn = nn.init.he_normal()

for module_name in model.state:
    if module_name != "memory":
        for module in getattr(model.state, module_name):
            for layer in getattr(module, "layers")[0]:
                if (layer["weight"].shape != (0,)):
                    layer["weight"] = init_fn(layer["weight"])


def add_gradients(g1, g2):
    """Recursively add two gradient structures (dicts, lists, or arrays)."""
    if isinstance(g1, dict):
        return {k: add_gradients(g1[k], g2[k]) for k in g1}
    elif isinstance(g1, list):
        return [add_gradients(x, y) for x, y in zip(g1, g2)]
    else:
        # Assume g1 and g2 are numeric arrays (or scalars)
        return g1 + g2

def scale_gradient(g, scale):
    """Recursively scale the gradient structure by a scalar."""
    if isinstance(g, dict):
        return {k: scale_gradient(g[k], scale) for k in g}
    elif isinstance(g, list):
        return [scale_gradient(x, scale) for x in g]
    else:
        return g * scale

#@partial(mx.compile, inputs=state, outputs=state)
def step(input, target, copy_len):
    loss_total = 0
    grad_accum = None  # will accumulate gradients from each time step
    grad_count = 0
    activation = None
    r = mx.zeros([network_param["memory_size"][1]]) * 1e-3
    w = mx.softmax(mx.ones([network_param["memory_size"][0]]))
    loss_grad_fn = nn.value_and_grad(model, loss_fn)

    for i, sequence in enumerate(input):
        logits, r, w, activation = model(sequence, r, w)
        if i > copy_len + 1:
            i_target = target[i - (copy_len + 1)]
            loss, grad_step = loss_grad_fn(logits, i_target)
            loss_total += loss
            optimizer.update(model, grad_step)

            # Accumulate the gradient.
            if grad_accum is None:
                grad_accum = grad_step  # Initialize with the first computed gradient.
            else:
                grad_accum = add_gradients(grad_accum, grad_step)
            grad_count += 1

    # Average the gradients over the number of update steps.
    if grad_count > 0:
        grad_accum = scale_gradient(grad_accum, 1.0 / grad_count)
    else:
        grad_accum = {}  # or however you want to handle "no gradient" cases.

    return loss_total / (copy_len + 1), grad_accum, activation


gradient_h = []
activation_h = []
loss_h = []

def train(iters, max_sequence_length, element_length):
    print("Theoretial init Loss ~.693")
    generator = train_pair_iter(max_sequence_length, element_length)
    for i in range(iters):
        input, target, copy_len = next(generator)
        loss, grad, activation = step(input, target, copy_len)
        gradient_h.append(grad)
        activation_h.append(activation)
        loss_h.append(loss)
        mx.eval(model.state, optimizer.state)
        if (i % 100 == 0): print(f'{i:<5.0f} loss_total: {loss:.4f}')
