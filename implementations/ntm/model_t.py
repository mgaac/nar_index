import mlx.core as mx
import mlx.nn as nn

class controller(nn.Module):
    def __init__(self, idim: int, odim: int, hdim: int, numl_shared: int, numl_con: int, numl_out: int, memory_size: list):
        super().__init__()

        self.memory = mx.ones([memory_size[0], memory_size[1]]) * 1e-6

        shared_layer_sizes = [idim + memory_size[1]] + [hdim] * numl_shared
        controller_layer_sizes = [hdim] * numl_con + [memory_size[0] * 3 + 3 * 2]  # a, e, k (m size) + shift + 3 addressing params.
        out_layer_sizes = [hdim] * numl_out + [odim]

        self.shared_layers = [nn.Sequential([nn.Linear(x, y, bias=False), nn.LayerNorm(0)]) for x, y in zip(shared_layer_sizes[:-1], shared_layer_sizes[1:])]
        self.controller_layers = [nn.Sequential([nn.Linear(x, y, bias=False), nn.LayerNorm(0)])  for x, y in zip(controller_layer_sizes[:-1], controller_layer_sizes[1:])]
        self.out_layers = [nn.Sequential([nn.Linear(x, y, bias=False), nn.LayerNorm(0)]) for x, y in zip(out_layer_sizes[:-1], out_layer_sizes[1:])]

    def __call__(self, x, r, wp):
        activations = {}

        def apply_layers(input, module, name):
            for i0, block in enumerate(module):
                for i1, layer in enumerate(block["layers"][0]):
                    input = mx.maximum(layer(input), 0)
                    activations[f"{name}_layer_{i0}.{i1}"] = input  # Store activations
            return input

        x = apply_layers(mx.concatenate([x, r]), self.shared_layers, "shared")
        x_out = apply_layers(x, self.out_layers, "output")
        x_ctrl = apply_layers(x, self.controller_layers, "controller")

        activations["final_output"] = x_out
        activations["final_controller"] = x_ctrl

        i, j = self.memory.shape
        a, e, k = x_ctrl[0:j * 3].reshape([3, j])
        s = x_ctrl[j * 3:j * 3 + 3]
        b, g, y = x_ctrl[j * 3 + 3:j * 3 + 6].reshape([3])

        # Ctrl parameter constraints
        g = mx.sigmoid(g)
        s = mx.softmax(s)
        y = 1 + y**2

        activations["g"] = g
        activations["s"] = s
        activations["y"] = y

        w = self.adressing(k, b, g, s, y, wp)
        r = self.read(w)
        self.write(w, a, e)

        activations["w"] = w
        activations["r"] = r

        return x_out, r, w, activations

    def adressing(self, k, b, g, s, y, wp):
        memory_norms = mx.linalg.norm(self.memory, axis=1) + 1e-8  # Avoid division by zero
        key_norm = mx.linalg.norm(k) + 1e-8
        similarity = (k @ self.memory.transpose()) / (key_norm * memory_norms)
        w = mx.softmax((b * similarity))

        w = g * w + ((1 - g) * wp)  # Interpolation
        w = mx.convolve(w, s, mode='same')  # Convolutional Shift
        w = mx.power(w, y) / mx.power(w, y).sum()  # Sharpening

        return w

    def read(self, w):
        return w @ self.memory

    def write(self, w, a, e):
        self.memory = self.memory * (1 - mx.outer(w, e))  # erase
        self.memory = self.memory + mx.outer(w, a)  # write

        return
