from model import controller

import mlx.optimizers as optim
import mlx.core as mx
import mlx.nn as nn

from functools import partial


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
    "l_rate": 1e-4
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

for module in model.state:
    if module != "memory":
        for layer_name in getattr(model.state, module):
            layer_name["weight"] = init_fn(layer_name["weight"])

@partial(mx.compile, inputs=state, outputs=state)
def step(input, target, copy_len):
    loss_total = 0.0
    r = mx.zeros([network_param["memory_size"][1]]) * 1e-3
    w = mx.random.normal([network_param["memory_size"][0]])
    loss_grad_fn = nn.value_and_grad(model, loss_fn)
    for i, sequence in enumerate(input):
        logits, r, w = model(sequence, r, w)
        if (i > copy_len + 1):
            i_target = target[i - (copy_len + 1)]
            loss, grad = loss_grad_fn(logits, i_target)
            loss_total += loss
            optimizer.update(model, grad)
    return loss_total / (copy_len + 1)

def train(iters, max_sequence_length, element_length):
    print("Theoretial init Loss ~.693")
    generator = train_pair_iter(max_sequence_length, element_length)
    for i in range(iters):
        input, target, copy_len = next(generator)
        loss = step(input, target, copy_len)
        mx.eval(model.state, optimizer.state)
        if (i % 100 == 0): print(f'{i:<5.0f} loss_total: {loss:.4f}')


train(300, 10, 5)
