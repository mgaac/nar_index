import mlx.core as mx
import mlx.nn as nn

class controller(nn.Module):
    def __init__(self, idim: int, odim: int, hdim: int, numl_shared: int, numl_con: int, numl_out: int, memory_size: list):
        super().__init__()

        shared_layer_sizes = [idim + memory_size[1]] + [hdim] * numl_shared
        controller_layer_sizes = [hdim] * numl_con + [memory_size[1] * 3 + 3 * 2]  # a, e, k (m size) + shift + 3 addressing params.
        out_layer_sizes = [hdim] * numl_out + [odim]

        self.shared_layers = [nn.Linear(x, y, bias=True) for x, y in zip(shared_layer_sizes[:-1], shared_layer_sizes[1:])]
        self.controller_layers = [nn.Linear(x, y, bias=True) for x, y in zip(controller_layer_sizes[:-1], controller_layer_sizes[1:])]
        self.out_layers = [nn.Linear(x, y, bias=True) for x, y in zip(out_layer_sizes[:-1], out_layer_sizes[1:])]

    def __call__(self, x, r, wp, memory):
        activations = {}

        def flatten_layers(module):
            layers = []
            for block in module:
                if hasattr(block, "layers"):
                    # If layers is a nested list (e.g. list-of-list), flatten it.
                    if isinstance(block.layers[0], list):
                        layers.extend(block.layers[0])
                    else:
                        layers.extend(block.layers)
                else:
                    layers.append(block)
            return layers

        def apply_layers(input, module, name, apply_final_activation=False):
            flat_layers = flatten_layers(module)
            for idx, layer in enumerate(flat_layers):
                # If this is the last layer and we don't want an activation, call layer directly.
                if idx == len(flat_layers) - 1 and not apply_final_activation:
                    input = layer(input)
                else:
                    preact = layer(input);
                    if idx != 0: preact += input
                    input = nn.silu(preact)
                activations[f"{name}_layer_{idx}"] = input
            return input

        x = apply_layers(mx.concatenate([x, r]), self.shared_layers, "shared", apply_final_activation=True)
        x_ctrl = apply_layers(x, self.controller_layers, "controller", apply_final_activation=False)
        x_out = apply_layers(x, self.out_layers, "output", apply_final_activation=False)

        activations["final_output"] = x_out
        activations["final_controller"] = x_ctrl

        i, j = memory.shape
        a, e, k = x_ctrl[0:j * 3].reshape([3, j])
        s = x_ctrl[j * 3:j * 3 + 3]
        b, g, y = x_ctrl[j * 3 + 3:j * 3 + 6].reshape([3])

        # Ctrl parameter constraints
        g = mx.sigmoid(g)
        s = mx.softmax(s)
        b = mx.log(mx.exp(b) + 1)
        y = mx.log(mx.exp(y) + 1) + 1
        a = mx.tanh(a)
        k = mx.tanh(k)
        e = mx.tanh(e)

        activations["g"] = g
        activations["s"] = s
        activations["y"] = y

        w = self.adressing(k, b, g, s, y, wp, memory)
        r = self.read(w, memory)
        new_memory = self.write(w, a, e, memory)

        activations["w"] = w
        activations["r"] = r

        return x_out, r, w, new_memory, activations

    def adressing(self, k, b, g, s, y, wp, memory):
        memory_norms = mx.linalg.norm(memory, axis=1) + 1e-8  # Avoid division by zero
        key_norm = mx.linalg.norm(k) + 1e-8
        similarity = (k @ memory.transpose()) / (key_norm * memory_norms)
        w = mx.softmax((b * similarity))

        w = g * w + ((1 - g) * wp)  # Interpolation
        w = mx.convolve(w, s, mode='same')  # Convolutional Shift
        w = mx.power(w, y) / mx.power(w, y).sum()  # Sharpening

        return mx.softmax(w)

    def read(self, w, memory):
        return w @ memory

    def write(self, w, a, e, memory):
        memory = memory * (1 - mx.outer(w, e))  # erase
        memory = memory + mx.outer(w, a)  # write

        return mx.sigmoid(memory)
