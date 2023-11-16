"""Simple network classes for Tensorflow based on tf.Module."""
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Linear(tf.Module):
    """A simple linear module.

  Always includes biases and only supports ReLU activations.
  """

    def __init__(self, in_size, out_size, activate_relu=True, name=None):
        if False:
            return 10
        'Creates a linear layer.\n\n    Args:\n      in_size: (int) number of inputs\n      out_size: (int) number of outputs\n      activate_relu: (bool) whether to include a ReLU activation layer\n      name: (string): the name to give to this layer\n    '
        super(Linear, self).__init__(name=name)
        self._activate_relu = activate_relu
        stddev = 1.0 / math.sqrt(in_size)
        self._weights = tf.Variable(tf.random.truncated_normal([in_size, out_size], mean=0.0, stddev=stddev), name='weights')
        self._bias = tf.Variable(tf.zeros([out_size]), name='bias')

    def __call__(self, tensor):
        if False:
            i = 10
            return i + 15
        y = tf.matmul(tensor, self._weights) + self._bias
        return tf.nn.relu(y) if self._activate_relu else y

class Sequential(tf.Module):
    """A simple sequential module.

  Always includes biases and only supports ReLU activations.
  """

    def __init__(self, layers, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Creates a model from successively applying layers.\n\n    Args:\n      layers: Iterable[tf.Module] that can be applied.\n      name: (string): the name to give to this layer\n    '
        super(Sequential, self).__init__(name=name)
        self._layers = layers

    def __call__(self, tensor):
        if False:
            while True:
                i = 10
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor

class MLP(tf.Module):
    """A simple dense network built from linear layers above."""

    def __init__(self, input_size, hidden_sizes, output_size, activate_final=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Create the MLP.\n\n    Args:\n      input_size: (int) number of inputs\n      hidden_sizes: (list) sizes (number of units) of each hidden layer\n      output_size: (int) number of outputs\n      activate_final: (bool) should final layer should include a ReLU\n      name: (string): the name to give to this network\n    '
        super(MLP, self).__init__(name=name)
        self._layers = []
        with self.name_scope:
            for size in hidden_sizes:
                self._layers.append(Linear(in_size=input_size, out_size=size))
                input_size = size
            self._layers.append(Linear(in_size=input_size, out_size=output_size, activate_relu=activate_final))

    @tf.Module.with_name_scope
    def __call__(self, x):
        if False:
            while True:
                i = 10
        for layer in self._layers:
            x = layer(x)
        return x

class MLPTorso(tf.Module):
    """A specialized half-MLP module when constructing multiple heads.

  Note that every layer includes a ReLU non-linearity activation.
  """

    def __init__(self, input_size, hidden_sizes, name=None):
        if False:
            i = 10
            return i + 15
        super(MLPTorso, self).__init__(name=name)
        self._layers = []
        with self.name_scope:
            for size in hidden_sizes:
                self._layers.append(Linear(in_size=input_size, out_size=size))
                input_size = size

    @tf.Module.with_name_scope
    def __call__(self, x):
        if False:
            print('Hello World!')
        for layer in self._layers:
            x = layer(x)
        return x