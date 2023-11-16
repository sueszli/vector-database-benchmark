"""
Title: Multi-GPU distributed training with JAX
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2023/07/11
Last modified: 2023/07/11
Description: Guide to multi-GPU/TPU training for Keras models with JAX.
Accelerator: GPU
"""
'\n## Introduction\n\nThere are generally two ways to distribute computation across multiple devices:\n\n**Data parallelism**, where a single model gets replicated on multiple devices or\nmultiple machines. Each of them processes different batches of data, then they merge\ntheir results. There exist many variants of this setup, that differ in how the different\nmodel replicas merge results, in whether they stay in sync at every batch or whether they\nare more loosely coupled, etc.\n\n**Model parallelism**, where different parts of a single model run on different devices,\nprocessing a single batch of data together. This works best with models that have a\nnaturally-parallel architecture, such as models that feature multiple branches.\n\nThis guide focuses on data parallelism, in particular **synchronous data parallelism**,\nwhere the different replicas of the model stay in sync after each batch they process.\nSynchronicity keeps the model convergence behavior identical to what you would see for\nsingle-device training.\n\nSpecifically, this guide teaches you how to use `jax.sharding` APIs to train Keras\nmodels, with minimal changes to your code, on multiple GPUs or TPUS (typically 2 to 16)\ninstalled on a single machine (single host, multi-device training). This is the\nmost common setup for researchers and small-scale industry workflows.\n'
"\n## Setup\n\nLet's start by defining the function that creates the model that we will train,\nand the function that creates the dataset we will train on (MNIST in this case).\n"
import os
os.environ['KERAS_BACKEND'] = 'jax'
import jax
import numpy as np
import tensorflow as tf
import keras
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

def get_model():
    if False:
        print('Hello World!')
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(filters=12, kernel_size=3, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters=24, kernel_size=6, use_bias=False, strides=2)(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=6, padding='same', strides=2, name='large_k')(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    return model

def get_datasets():
    if False:
        print('Hello World!')
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    eval_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return (train_data, eval_data)
"\n## Single-host, multi-device synchronous training\n\nIn this setup, you have one machine with several GPUs or TPUs on it (typically 2 to 16).\nEach device will run a copy of your model (called a **replica**). For simplicity, in\nwhat follows, we'll assume we're dealing with 8 GPUs, at no loss of generality.\n\n**How it works**\n\nAt each step of training:\n\n- The current batch of data (called **global batch**) is split into 8 different\n  sub-batches (called **local batches**). For instance, if the global batch has 512\n  samples, each of the 8 local batches will have 64 samples.\n- Each of the 8 replicas independently processes a local batch: they run a forward pass,\n  then a backward pass, outputting the gradient of the weights with respect to the loss of\n  the model on the local batch.\n- The weight updates originating from local gradients are efficiently merged across the 8\n  replicas. Because this is done at the end of every step, the replicas always stay in\n  sync.\n\nIn practice, the process of synchronously updating the weights of the model replicas is\nhandled at the level of each individual weight variable. This is done through a using\na `jax.sharding.NamedSharding` that is configured to replicate the variables.\n\n**How to use it**\n\nTo do single-host, multi-device synchronous training with a Keras model, you\nwould use the `jax.sharding` features. Here's how it works:\n\n- We first create a device mesh using `mesh_utils.create_device_mesh`.\n- We use `jax.sharding.Mesh`, `jax.sharding.NamedSharding` and\n  `jax.sharding.PartitionSpec` to define how to partition JAX arrays.\n    - We specify that we want to replicate the model and optimizer variables\n      across all devices by using a spec with no axis.\n    - We specify that we want to shard the data across devices by using a spec\n      that splits along the batch dimension.\n- We use `jax.device_put` to replicate the model and optimizer variables across\n  devices. This happens once at the beginning.\n- In the training loop, for each batch that we process, we use `jax.device_put`\n  to split the batch across devices before invoking the train step.\n\nHere's the flow, where each step is split into its own utility function:\n"
num_epochs = 2
batch_size = 64
(train_data, eval_data) = get_datasets()
train_data = train_data.batch(batch_size, drop_remainder=True)
model = get_model()
optimizer = keras.optimizers.Adam(0.001)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
(one_batch, one_batch_labels) = next(iter(train_data))
model.build(one_batch)
optimizer.build(model.trainable_variables)

def compute_loss(trainable_variables, non_trainable_variables, x, y):
    if False:
        while True:
            i = 10
    (y_pred, updated_non_trainable_variables) = model.stateless_call(trainable_variables, non_trainable_variables, x)
    loss_value = loss(y, y_pred)
    return (loss_value, updated_non_trainable_variables)
compute_gradients = jax.value_and_grad(compute_loss, has_aux=True)

@jax.jit
def train_step(train_state, x, y):
    if False:
        print('Hello World!')
    (trainable_variables, non_trainable_variables, optimizer_variables) = train_state
    ((loss_value, non_trainable_variables), grads) = compute_gradients(trainable_variables, non_trainable_variables, x, y)
    (trainable_variables, optimizer_variables) = optimizer.stateless_apply(optimizer_variables, grads, trainable_variables)
    return (loss_value, (trainable_variables, non_trainable_variables, optimizer_variables))

def get_replicated_train_state(devices):
    if False:
        i = 10
        return i + 15
    var_mesh = Mesh(devices, axis_names='_')
    var_replication = NamedSharding(var_mesh, P())
    trainable_variables = jax.device_put(model.trainable_variables, var_replication)
    non_trainable_variables = jax.device_put(model.non_trainable_variables, var_replication)
    optimizer_variables = jax.device_put(optimizer.variables, var_replication)
    return (trainable_variables, non_trainable_variables, optimizer_variables)
num_devices = len(jax.local_devices())
print(f'Running on {num_devices} devices: {jax.local_devices()}')
devices = mesh_utils.create_device_mesh((num_devices,))
data_mesh = Mesh(devices, axis_names=('batch',))
data_sharding = NamedSharding(data_mesh, P('batch'))
(x, y) = next(iter(train_data))
sharded_x = jax.device_put(x.numpy(), data_sharding)
print('Data sharding')
jax.debug.visualize_array_sharding(jax.numpy.reshape(sharded_x, [-1, 28 * 28]))
train_state = get_replicated_train_state(devices)
for epoch in range(num_epochs):
    data_iter = iter(train_data)
    for data in data_iter:
        (x, y) = data
        sharded_x = jax.device_put(x.numpy(), data_sharding)
        (loss_value, train_state) = train_step(train_state, sharded_x, y.numpy())
    print('Epoch', epoch, 'loss:', loss_value)
(trainable_variables, non_trainable_variables, optimizer_variables) = train_state
for (variable, value) in zip(model.trainable_variables, trainable_variables):
    variable.assign(value)
for (variable, value) in zip(model.non_trainable_variables, non_trainable_variables):
    variable.assign(value)
"\nThat's it!\n"