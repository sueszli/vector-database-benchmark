"""
Title: Making new layers and models via subclassing
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2019/03/01
Last modified: 2023/06/25
Description: Complete guide to writing `Layer` and `Model` objects from scratch.
Accelerator: None
"""
"\n## Introduction\n\nThis guide will cover everything you need to know to build your own\nsubclassed layers and models. In particular, you'll learn about the following features:\n\n- The `Layer` class\n- The `add_weight()` method\n- Trainable and non-trainable weights\n- The `build()` method\n- Making sure your layers can be used with any backend\n- The `add_loss()` method\n- The `training` argument in `call()`\n- The `mask` argument in `call()`\n- Making sure your layers can be serialized\n\nLet's dive in.\n"
'\n## Setup\n'
import numpy as np
import keras
from keras import ops
from keras import layers
'\n## The `Layer` class: the combination of state (weights) and some computation\n\nOne of the central abstractions in Keras is the `Layer` class. A layer\nencapsulates both a state (the layer\'s "weights") and a transformation from\ninputs to outputs (a "call", the layer\'s forward pass).\n\nHere\'s a densely-connected layer. It has two state variables:\nthe variables `w` and `b`.\n'

class Linear(keras.layers.Layer):

    def __init__(self, units=32, input_dim=32):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.w = self.add_weight(shape=(input_dim, units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(units,), initializer='zeros', trainable=True)

    def call(self, inputs):
        if False:
            print('Hello World!')
        return ops.matmul(inputs, self.w) + self.b
'\nYou would use a layer by calling it on some tensor input(s), much like a Python\nfunction.\n'
x = ops.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
'\nNote that the weights `w` and `b` are automatically tracked by the layer upon\nbeing set as layer attributes:\n'
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
"\n## Layers can have non-trainable weights\n\nBesides trainable weights, you can add non-trainable weights to a layer as\nwell. Such weights are meant not to be taken into account during\nbackpropagation, when you are training the layer.\n\nHere's how to add and use a non-trainable weight:\n"

class ComputeSum(keras.layers.Layer):

    def __init__(self, input_dim):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.total = self.add_weight(initializer='zeros', shape=(input_dim,), trainable=False)

    def call(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        self.total.assign_add(ops.sum(inputs, axis=0))
        return self.total
x = ops.ones((2, 2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())
"\nIt's part of `layer.weights`, but it gets categorized as a non-trainable weight:\n"
print('weights:', len(my_sum.weights))
print('non-trainable weights:', len(my_sum.non_trainable_weights))
print('trainable_weights:', my_sum.trainable_weights)
'\n## Best practice: deferring weight creation until the shape of the inputs is known\n\nOur `Linear` layer above took an `input_dim` argument that was used to compute\nthe shape of the weights `w` and `b` in `__init__()`:\n'

class Linear(keras.layers.Layer):

    def __init__(self, units=32, input_dim=32):
        if False:
            while True:
                i = 10
        super().__init__()
        self.w = self.add_weight(shape=(input_dim, units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(units,), initializer='zeros', trainable=True)

    def call(self, inputs):
        if False:
            return 10
        return ops.matmul(inputs, self.w) + self.b
'\nIn many cases, you may not know in advance the size of your inputs, and you\nwould like to lazily create weights when that value becomes known, some time\nafter instantiating the layer.\n\nIn the Keras API, we recommend creating layer weights in the\n`build(self, inputs_shape)` method of your layer. Like this:\n'

class Linear(keras.layers.Layer):

    def __init__(self, units=32):
        if False:
            return 10
        super().__init__()
        self.units = units

    def build(self, input_shape):
        if False:
            while True:
                i = 10
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        return ops.matmul(inputs, self.w) + self.b
"\nThe `__call__()` method of your layer will automatically run build the first time\nit is called. You now have a layer that's lazy and thus easier to use:\n"
linear_layer = Linear(32)
y = linear_layer(x)
'\nImplementing `build()` separately as shown above nicely separates creating weights\nonly once from using weights in every call.\n'
'\n## Layers are recursively composable\n\nIf you assign a Layer instance as an attribute of another Layer, the outer layer\nwill start tracking the weights created by the inner layer.\n\nWe recommend creating such sublayers in the `__init__()` method and leave it to\nthe first `__call__()` to trigger building their weights.\n'

class MLPBlock(keras.layers.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        x = self.linear_1(inputs)
        x = keras.activations.relu(x)
        x = self.linear_2(x)
        x = keras.activations.relu(x)
        return self.linear_3(x)
mlp = MLPBlock()
y = mlp(ops.ones(shape=(3, 64)))
print('weights:', len(mlp.weights))
print('trainable weights:', len(mlp.trainable_weights))
"\n## Backend-agnostic layers and backend-specific layers\n\nAs long as a layer only uses APIs from the `keras.ops` namespace\n(or other Keras namespaces such as `keras.activations`, `keras.random`, or `keras.layers`),\nthen it can be used with any backend -- TensorFlow, JAX, or PyTorch.\n\nAll layers you've seen so far in this guide work with all Keras backends.\n\nThe `keras.ops` namespace gives you access to:\n\n- The NumPy API, e.g. `ops.matmul`, `ops.sum`, `ops.reshape`, `ops.stack`, etc.\n- Neural networks-specific APIs such as `ops.softmax`, `ops.conv`, `ops.binary_crossentropy`, `ops.relu`, etc.\n\nYou can also use backend-native APIs in your layers (such as `tf.nn` functions),\nbut if you do this, then your layer will only be usable with the backend in question.\nFor instance, you could write the following JAX-specific layer using `jax.numpy`:\n\n```python\nimport jax\n\nclass Linear(keras.layers.Layer):\n    ...\n\n    def call(self, inputs):\n        return jax.numpy.matmul(inputs, self.w) + self.b\n```\n\nThis would be the equivalent TensorFlow-specific layer:\n\n```python\nimport tensorflow as tf\n\nclass Linear(keras.layers.Layer):\n    ...\n\n    def call(self, inputs):\n        return tf.matmul(inputs, self.w) + self.b\n```\n\nAnd this would be the equivalent PyTorch-specific layer:\n\n```python\nimport torch\n\nclass Linear(keras.layers.Layer):\n    ...\n\n    def call(self, inputs):\n        return torch.matmul(inputs, self.w) + self.b\n```\n\nBecause cross-backend compatibility is a tremendously useful property, we strongly\nrecommend that you seek to always make your layers backend-agnostic by leveraging\nonly Keras APIs.\n"
'\n## The `add_loss()` method\n\nWhen writing the `call()` method of a layer, you can create loss tensors that\nyou will want to use later, when writing your training loop. This is doable by\ncalling `self.add_loss(value)`:\n'

class ActivityRegularizationLayer(keras.layers.Layer):

    def __init__(self, rate=0.01):
        if False:
            while True:
                i = 10
        super().__init__()
        self.rate = rate

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        self.add_loss(self.rate * ops.mean(inputs))
        return inputs
'\nThese losses (including those created by any inner layer) can be retrieved via\n`layer.losses`. This property is reset at the start of every `__call__()` to\nthe top-level layer, so that `layer.losses` always contains the loss values\ncreated during the last forward pass.\n'

class OuterLayer(keras.layers.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.activity_reg = ActivityRegularizationLayer(0.01)

    def call(self, inputs):
        if False:
            while True:
                i = 10
        return self.activity_reg(inputs)
layer = OuterLayer()
assert len(layer.losses) == 0
_ = layer(ops.zeros((1, 1)))
assert len(layer.losses) == 1
_ = layer(ops.zeros((1, 1)))
assert len(layer.losses) == 1
'\nIn addition, the `loss` property also contains regularization losses created\nfor the weights of any inner layer:\n'

class OuterLayerWithKernelRegularizer(keras.layers.Layer):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.dense = keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.001))

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        return self.dense(inputs)
layer = OuterLayerWithKernelRegularizer()
_ = layer(ops.zeros((1, 1)))
print(layer.losses)
'\nThese losses are meant to be taken into account when writing custom training loops.\n\nThey also work seamlessly with `fit()` (they get automatically summed and added to the main loss, if any):\n'
inputs = keras.Input(shape=(3,))
outputs = ActivityRegularizationLayer()(inputs)
model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))
model.compile(optimizer='adam')
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))
'\n## You can optionally enable serialization on your layers\n\nIf you need your custom layers to be serializable as part of a\n[Functional model](/guides/functional_api/), you can optionally implement a `get_config()`\nmethod:\n'

class Linear(keras.layers.Layer):

    def __init__(self, units=32):
        if False:
            print('Hello World!')
        super().__init__()
        self.units = units

    def build(self, input_shape):
        if False:
            while True:
                i = 10
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        if False:
            return 10
        return ops.matmul(inputs, self.w) + self.b

    def get_config(self):
        if False:
            return 10
        return {'units': self.units}
layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
"\nNote that the `__init__()` method of the base `Layer` class takes some keyword\narguments, in particular a `name` and a `dtype`. It's good practice to pass\nthese arguments to the parent class in `__init__()` and to include them in the\nlayer config:\n"

class Linear(keras.layers.Layer):

    def __init__(self, units=32, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        if False:
            for i in range(10):
                print('nop')
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        if False:
            return 10
        return ops.matmul(inputs, self.w) + self.b

    def get_config(self):
        if False:
            return 10
        config = super().get_config()
        config.update({'units': self.units})
        return config
layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
'\nIf you need more flexibility when deserializing the layer from its config, you\ncan also override the `from_config()` class method. This is the base\nimplementation of `from_config()`:\n\n```python\ndef from_config(cls, config):\n    return cls(**config)\n```\n\nTo learn more about serialization and saving, see the complete\n[guide to saving and serializing models](/guides/serialization_and_saving/).\n'
'\n## Privileged `training` argument in the `call()` method\n\nSome layers, in particular the `BatchNormalization` layer and the `Dropout`\nlayer, have different behaviors during training and inference. For such\nlayers, it is standard practice to expose a `training` (boolean) argument in\nthe `call()` method.\n\nBy exposing this argument in `call()`, you enable the built-in training and\nevaluation loops (e.g. `fit()`) to correctly use the layer in training and\ninference.\n'

class CustomDropout(keras.layers.Layer):

    def __init__(self, rate, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if False:
            i = 10
            return i + 15
        if training:
            return keras.random.dropout(inputs, rate=self.rate)
        return inputs
'\n## Privileged `mask` argument in the `call()` method\n\nThe other privileged argument supported by `call()` is the `mask` argument.\n\nYou will find it in all Keras RNN layers. A mask is a boolean tensor (one\nboolean value per timestep in the input) used to skip certain input timesteps\nwhen processing timeseries data.\n\nKeras will automatically pass the correct `mask` argument to `__call__()` for\nlayers that support it, when a mask is generated by a prior layer.\nMask-generating layers are the `Embedding`\nlayer configured with `mask_zero=True`, and the `Masking` layer.\n'
'\n## The `Model` class\n\nIn general, you will use the `Layer` class to define inner computation blocks,\nand will use the `Model` class to define the outer model -- the object you\nwill train.\n\nFor instance, in a ResNet50 model, you would have several ResNet blocks\nsubclassing `Layer`, and a single `Model` encompassing the entire ResNet50\nnetwork.\n\nThe `Model` class has the same API as `Layer`, with the following differences:\n\n- It exposes built-in training, evaluation, and prediction loops\n(`model.fit()`, `model.evaluate()`, `model.predict()`).\n- It exposes the list of its inner layers, via the `model.layers` property.\n- It exposes saving and serialization APIs (`save()`, `save_weights()`...)\n\nEffectively, the `Layer` class corresponds to what we refer to in the\nliterature as a "layer" (as in "convolution layer" or "recurrent layer") or as\na "block" (as in "ResNet block" or "Inception block").\n\nMeanwhile, the `Model` class corresponds to what is referred to in the\nliterature as a "model" (as in "deep learning model") or as a "network" (as in\n"deep neural network").\n\nSo if you\'re wondering, "should I use the `Layer` class or the `Model` class?",\nask yourself: will I need to call `fit()` on it? Will I need to call `save()`\non it? If so, go with `Model`. If not (either because your class is just a block\nin a bigger system, or because you are writing training & saving code yourself),\nuse `Layer`.\n\nFor instance, we could take our mini-resnet example above, and use it to build\na `Model` that we could train with `fit()`, and that we could save with\n`save_weights()`:\n'
'\n```python\nclass ResNet(keras.Model):\n\n    def __init__(self, num_classes=1000):\n        super().__init__()\n        self.block_1 = ResNetBlock()\n        self.block_2 = ResNetBlock()\n        self.global_pool = layers.GlobalAveragePooling2D()\n        self.classifier = Dense(num_classes)\n\n    def call(self, inputs):\n        x = self.block_1(inputs)\n        x = self.block_2(x)\n        x = self.global_pool(x)\n        return self.classifier(x)\n\n\nresnet = ResNet()\ndataset = ...\nresnet.fit(dataset, epochs=10)\nresnet.save(filepath.keras)\n```\n'
"\n## Putting it all together: an end-to-end example\n\nHere's what you've learned so far:\n\n- A `Layer` encapsulate a state (created in `__init__()` or `build()`) and some\ncomputation (defined in `call()`).\n- Layers can be recursively nested to create new, bigger computation blocks.\n- Layers are backend-agnostic as long as they only use Keras APIs. You can use\nbackend-native APIs (such as `jax.numpy`, `torch.nn` or `tf.nn`), but then\nyour layer will only be usable with that specific backend.\n- Layers can create and track losses (typically regularization losses)\nvia `add_loss()`.\n- The outer container, the thing you want to train, is a `Model`. A `Model` is\njust like a `Layer`, but with added training and serialization utilities.\n\nLet's put all of these things together into an end-to-end example: we're going\nto implement a Variational AutoEncoder (VAE) in a backend-agnostic fashion\n-- so that it runs the same with TensorFlow, JAX, and PyTorch.\nWe'll train it on MNIST digits.\n\nOur VAE will be a subclass of `Model`, built as a nested composition of layers\nthat subclass `Layer`. It will feature a regularization loss (KL divergence).\n"

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        if False:
            while True:
                i = 10
        (z_mean, z_log_var) = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name='encoder', **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        if False:
            print('Hello World!')
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return (z_mean, z_log_var, z)

class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, name='decoder', **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        x = self.dense_proj(inputs)
        return self.dense_output(x)

class VariationalAutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, original_dim, intermediate_dim=64, latent_dim=32, name='autoencoder', **kwargs):
        if False:
            return 10
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        (z_mean, z_log_var, z) = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * ops.mean(z_log_var - ops.square(z_mean) - ops.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed
"\nLet's train it on MNIST using the `fit()` API:\n"
((x_train, _), _) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
original_dim = 784
vae = VariationalAutoEncoder(784, 64, 32)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
vae.compile(optimizer, loss=keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=2, batch_size=64)