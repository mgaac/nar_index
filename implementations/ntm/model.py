import mlx.core as mx
import mlx.nn as nn

class controller(nn.Module):
    def __init__ (self, idim: int, odim: int, hdim: int, numl_shared: int, numl_con: int, numl_out: int, memory_size: list):
       super().__init__()

       self.memory = mx.random.normal([memory_size[0], memory_size[1]])

       shared_layer_sizes = [idim + memory_size[1]] + [hdim] * numl_shared
       controller_layer_sizes = [hdim] * numl_con + [memory_size[0] * 3 + 3 * 2] # a, e, k (m size) + shift + 3 addressing params.
       out_layer_sizes = [hdim] * numl_out + [odim]

       self.shared_layers = [nn.Linear(x, y) for x, y in zip(shared_layer_sizes[:-1], shared_layer_sizes[1:])]
       self.controller_layers = [nn.Linear(x, y) for x, y in zip(controller_layer_sizes[:-1], controller_layer_sizes[1:])]
       self.out_layers = [nn.Linear(x, y) for x, y in zip(out_layer_sizes[:-1], out_layer_sizes[1:])]

    def __call__(self, x, r, wp):
        def apply_layers(input, layers):
            for i, layer in enumerate(layers):
                input = mx.maximum(layer(input), 0)
            return input

        x = apply_layers(mx.concatenate([x, r]), self.shared_layers)
        x_out = mx.sigmoid(apply_layers(x, self.out_layers))
        x_ctrl = apply_layers(x, self.controller_layers)

        i, j = self.memory.shape
        a, e, k = x_ctrl[0:j * 3].reshape([3, j])
        s = x_ctrl[j * 3:j * 3 + 3]
        b, g, y = x_ctrl[j * 3 + 3:j * 3 + 6].reshape([3])

        w = self.adressing(k, b, g, s, y, wp)
        r = self.read(w)
        self.write(w, a, e)

        return x_out, r, w

    def adressing(self, k, b, g, s, y, wp):
        similarity = (k @ self.memory.transpose()) / (mx.linalg.norm(k) * mx.linalg.norm(self.memory, axis=1))
        w = (b * (similarity)).exp() / (b * (similarity)).exp().sum() # Focusing by Content
        w = g * w + ((1 - g) *  wp)  # Interpolation
        w = mx.convolve(w, s, mode='same') # Convolutional Shift
        w = mx.power(w, y) / mx.power(w, y).sum() # Sharpening

        return w

    def read(self, w):
        r = w @ self.memory

        return r

    def write(self, w, a, e):
        self.memory = self.memory * (1 - mx.outer(w, e)) # erase
        self.memory = self.memory + mx.outer(w, a) # write

        return


idim = 10
odim = 10
hdim = 128
numl_shared = 10
numl_con = 10
numl_out = 10
memory_size = [64, 10]

model = controller(idim, odim, hdim, numl_shared, numl_con, numl_out, memory_size)

x = mx.random.normal([10])
r = mx.random.normal([10])
w = mx.random.normal([64])

model(x, r, w)
