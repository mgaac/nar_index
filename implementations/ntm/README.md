# **Neural Turing Machines**
- - -
An implementation of *Neural Turing Machines* in **mlx** as described by arXiv:1410.5401v2.
> We extend the capabilities of neural networks by coupling them to external memory resources, which they can interact with by attentional processes. The combined system is analogous to a Turing Machine or Von Neumann architecture but is differentiable end-to-end, allowing it to be efficiently trained with gradient descent. Preliminary results demonstrate that Neural Turing Machines can infer simple algorithms such as copying, sorting, and associative recall from input and output examples.



More than anything, I am now convinced, NTMs speak to the will of their creators to somehow manage the ungodly gradient dynamics that emerge when training them, especially when dealing with the feed-forward variant—a challenge worsened by the lack of open-source implementations and, more surprisingly, by the absence of substantial literature investigating them. Many hours were spent, nonetheless, modifying the architecture and training loop in the hopes replicating the original paper’s findings. This aim, however, was not achieved—partly due to serendipitous gradient collapses to NaN, and partly due to insufficient access to compute resources. It does feel odd, however, that not much is spoken about them, specially given how intutive they seem as a natural extension of regular neural networks.
- - -


### Implementation Details

The architecture implements a Neural Turing Machine (NTM) that integrates a feed-forward controller with an external memory module. The design is divided into three main parts: shared layers, separate controller and output layers, and explicit memory interactions.


#### **Feed-Forward Controller**

#### **Shared Layers**

- **Input Composition:**
  The shared layers receive a concatenated input composed of:
  - The raw input vector.
  - A read vector extracted from the external memory.

- **Layer Configuration:**
  The shared layers are built using a series of linear transformations defined by:
  ```python
  shared_layer_sizes = [idim + memory_size[1]] + [hdim] * numl_shared
  self.shared_layers = [nn.Linear(x, y, bias=True) for x, y in zip(shared_layer_sizes[:-1], shared_layer_sizes[1:])]
  ```
  This design allows the model to process the combined features into a high-dimensional representation.

#### **Controller Layers**

- **Purpose:**
  The controller layers further transform the shared representation to produce parameters necessary for memory operations.

- **Layer Structure:**
  The architecture uses a feed-forward network whose final output is split into several groups controlling different aspects of memory manipulation:
  ```python
  controller_layer_sizes = [hdim] * numl_con + [memory_size[1] * 3 + 3 * 2]
  self.controller_layers = [nn.Linear(x, y, bias=True) for x, y in zip(controller_layer_sizes[:-1], controller_layer_sizes[1:])]
  ```
  - The final layer outputs values that are later segmented into parameters such as:
    - **Write vectors (`a` and `e`)** for the write operation.
    - **Key (`k`)** for content-based addressing.
    - **Shift parameters (`s`)** for location-based addressing.
    - **Additional scalars (`b`, `g`, `y`)** for bias, interpolation, and sharpening, respectively.

#### **Output Layers**

- **Purpose:**
  Parallel to the controller path, the output layers produce the final prediction of the network.

- **Layer Structure:**
  They are defined as:
  ```python
  out_layer_sizes = [hdim] * numl_out + [odim]
  self.out_layers = [nn.Linear(x, y, bias=True) for x, y in zip(out_layer_sizes[:-1], out_layer_sizes[1:])]
  ```
  This branch processes the shared representation into the final output dimension.

- - -

#### **Memory Interactions**

#### **Addressing Mechanism**

The addressing function computes how the controller interacts with the external memory using a combination of content-based and location-based addressing.

- **Content-Based Addressing:**
  - **Similarity Calculation:**
    The model computes cosine similarity between a key vector `k` and memory slots:
    ```python
    similarity = (k @ memory.transpose()) / (key_norm * memory_norms)
    ```
  - **Bias and Softmax:**
    Similarity scores are scaled by a bias parameter `b` and normalized using softmax:
    ```python
    w = mx.softmax((b * similarity))
    ```

- **Interpolation and Shifting:**
  - **Interpolation:**
    The address vector is interpolated with a previous weight vector `wp` using a gating parameter `g`:
    ```python
    w = g * w + ((1 - g) * wp)
    ```
  - **Convolutional Shift:**
    A convolution operation with kernel `s` shifts the weights:
    ```python
    w = mx.convolve(w, s, mode='same')
    ```
  - **Sharpening:**
    The weights are then sharpened using a power operation parameterized by `y`:
    ```python
    w = mx.power(w, y) / mx.power(w, y).sum()
    ```
  - **Final Normalization:**
    A final softmax is applied to ensure proper normalization.

#### **Read Operation**

The read operation extracts a weighted sum of the memory slots based on the addressing weights:
```python
def read(self, w, memory):
    return w @ memory
```
This mechanism produces a read vector that is used as part of the shared input in subsequent time steps.

#### **Write Operation**

The write operation updates the memory through a two-step process:
- **Erase:**
  The memory is first partially erased based on an erase vector `e`:
  ```python
  memory = memory * (1 - mx.outer(w, e))
  ```
- **Add:**
  Then, the memory is updated by adding new information defined by a write vector `a`:
  ```python
  memory = memory + mx.outer(w, a)
  ```
- **Activation:**
  A sigmoid function is applied to bound the updated memory values:
  ```python
  return mx.sigmoid(memory)
  ```
