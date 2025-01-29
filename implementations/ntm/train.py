from model import controller
from functools import partial
import mlx.optimizers as optim
import mlx.core as mx
import mlx.nn as nn


network_param = {
    "idim": 10,
    "odim": 10,
    "hdim": 128,
    "numl_shared": 3,
    "numl_con": 3,
    "numl_out": 3,
    "memory_size": [64, 10]
}

train_param = {
    "l_rate": .01
}

def loss_fn(logits, targets):
    return nn.losses.cross_entropy(logits, targets)

def eval_fn(logits, targets):
    return mx.sum(1 - (logits - targets)) / logits.shape[0]

def train_pair_iter(max_sequence_length, element_length):
    while(True):
        copy_len = mx.random.randint(1, max_sequence_length)
        target = mx.random.randint(0, 2, shape=[copy_len, element_length])
        input = mx.concatenate([target, mx.ones([1, element_length])])
        input = mx.concatenate([input, mx.zeros([copy_len, element_length])])
        yield input, target, copy_len.item()

model = controller(**network_param)
optimizer = optim.SGD(learning_rate=train_param["l_rate"])
state = [model.state, optimizer.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(input, target, copy_len):
    loss_total = 0.0
    r = mx.zeros([network_param["memory_size"][1]])
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
    generator = train_pair_iter(max_sequence_length, element_length)
    for i in range(iters):
        input, target, copy_len = next(generator)
        loss = step(input, target, copy_len)
        mx.eval(model.state, optimizer.state)
        print(loss)


train(10, 10, 10)
