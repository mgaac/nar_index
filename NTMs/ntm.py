import mlx.core as mx
import mlx.nn as nn


class controller(nn.Moduel):
    def __init__ (self, idim: int, hdim: int, numl_shared: int, numl_con: int, numl_out: int, nl:int , odim: int, memory_size: list):
       super().__init__()

       self.memory = mx.random.normal(memory_size[0], memory_size[1])

       shared_layer_sizes = [idim + memory_size[5]] + [hdim] * numl_shared
       controller_layer_sizes = [hdim] * numl_con + [4 + memory_size[1] * 3] # a, e, k (m size) + 4 addressing params.
       out_layer_sizes = [hdim] * numl_out + [odim]

       self.shared_layers = [nn.Linear(x, y) for x, y in zip(shared_layer_sizes[:-1], shared_layer_sizes[1:])]
       self.controller_layers = [nn.Linear(x, y) for x, y in zip(controller_layer_sizes[:-1], controller_layer_sizes[1:])]
       self.out_layers = [nn.Linear(x, y) for x, y in zip(out_layer_sizes[:-1], out_layer_sizes[1:])]

    def __call__(self, x, r, wp):
        def apply_layers(input, layers):
            for i, layer in enumerate(layers):
                input = mx.maximum(layer(input), 0)
            return input

        x = apply_layers(x + r, self.shared_layers)
        x_out = apply_layers(x, self.out_layers)
        x_ctrl = apply_layers(x, self.controller_layers)

        sizes = [self.memory_size[1]] * 3 + [4]
        ctrl = [x_ctrl[sum(sizes[:i]):sum(sizes[:i+1])] for i, _ in enumerate(sizes)]

        a, e, k = ctrl[:3]
        b, g, s, y = ctrl[3]

        w = self.adressing(k,b,g,s,y, wp)
        r = self.read(w)
        self.write(w, a, e)

        return x_out, r, w

    def adressing(self, k, b, g, s, y, wp):
        similarity = (k @ self.memory.transpose()) / (mx.linalg.norm(k) * mx.linalg.norm(self.memory, axis=1))
        w = (b * (similarity)).exp() / (b * (similarity)).exp().sum() # Focusing by Content
        w = g * w + ((1 - g) * wp)  # Interpolation
        w = mx.convolve(w, s, mode='same') # Convolutional Shift
        w = mx.power(w, y) / mx.power(w, y).sum() # Sharpening

        return w

    def read(self, w):
        r = w @ self.memory

        return r

    def write(self, w, a, e):
        self.memory = self.memory * (mx.ones(1, w.shape[1]) - w * e) # erase
        self.memory = self.memory + w * a # write

        return
