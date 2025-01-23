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

    def __call__(self, x):
       return x

    def adressing(self, k, b, g, s, y, wp):

        similarity = (k @ self.memory.transpose()) / (mx.linalg.norm(k) * mx.linalg.norm(self.memory, axis=1))
        w = (b * (similarity)).exp() / (b * (similarity)).exp().sum() # Focusing by Content

        w = g * w + ((1-g) * wp)  # Interpolation

        w = mx.convolve(w, s, mode='same') # Convolutional Shift


        # Sharpening
       return w

    def read(self, w):
        return r

    def write(self, w, a, e):
        return

z = [10] + [2] * 4
for x in zip(z[:-1], z[1:]):
    print(x)
