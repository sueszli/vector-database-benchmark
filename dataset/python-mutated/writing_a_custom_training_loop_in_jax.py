"""
Title: Writing a training loop from scratch in JAX
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2023/06/25
Last modified: 2023/06/25
Description: Writing low-level training & evaluation loops in JAX.
Accelerator: None
"""
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'jax'
import jax
import tensorflow as tf
import keras
import numpy as np
'\n## Introduction\n\nKeras provides default training and evaluation loops, `fit()` and `evaluate()`.\nTheir usage is covered in the guide\n[Training & evaluation with the built-in methods](https://keras.io/guides/training_with_built_in_methods/).\n\nIf you want to customize the learning algorithm of your model while still leveraging\nthe convenience of `fit()`\n(for instance, to train a GAN using `fit()`), you can subclass the `Model` class and\nimplement your own `train_step()` method, which\nis called repeatedly during `fit()`.\n\nNow, if you want very low-level control over training & evaluation, you should write\nyour own training & evaluation loops from scratch. This is what this guide is about.\n'
"\n## A first end-to-end example\n\nTo write a custom training loop, we need the following ingredients:\n\n- A model to train, of course.\n- An optimizer. You could either use an optimizer from `keras.optimizers`, or\none from the `optax` package.\n- A loss function.\n- A dataset. The standard in the JAX ecosystem is to load data via `tf.data`,\nso that's what we'll use.\n\nLet's line them up.\n\nFirst, let's get the model and the MNIST dataset:\n"

def get_model():
    if False:
        return 10
    inputs = keras.Input(shape=(784,), name='digits')
    x1 = keras.layers.Dense(64, activation='relu')(inputs)
    x2 = keras.layers.Dense(64, activation='relu')(x1)
    outputs = keras.layers.Dense(10, name='predictions')(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
model = get_model()
batch_size = 32
((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784)).astype('float32')
x_test = np.reshape(x_test, (-1, 784)).astype('float32')
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)
"\nNext, here's the loss function and the optimizer.\nWe'll use a Keras optimizer in this case.\n"
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
"\n### Getting gradients in JAX \n\nLet's train our model using mini-batch gradient with a custom training loop.\n\nIn JAX, gradients are computed via *metaprogramming*: you call the `jax.grad` (or\n`jax.value_and_grad` on a function in order to create a gradient-computing function\nfor that first function.\n\nSo the first thing we need is a function that returns the loss value.\nThat's the function we'll use to generate the gradient function. Something like this:\n\n```python\ndef compute_loss(x, y):\n    ...\n    return loss\n```\n\nOnce you have such a function, you can compute gradients via metaprogramming as such:\n\n```python\ngrad_fn = jax.grad(compute_loss)\ngrads = grad_fn(x, y)\n```\n\nTypically, you don't just want to get the gradient values, you also want to get\nthe loss value. You can do this by using `jax.value_and_grad` instead of `jax.grad`:\n\n```python\ngrad_fn = jax.value_and_grad(compute_loss)\nloss, grads = grad_fn(x, y)\n```\n\n### JAX computation is purely stateless\n\nIn JAX, everything must be a stateless function -- so our loss computation function\nmust be stateless as well. That means that all Keras variables (e.g. weight tensors)\nmust be passed as function inputs, and any variable that has been updated during the\nforward pass must be returned as function output. The function have no side effect.\n\nDuring the forward pass, the non-trainable variables of a Keras model might get\nupdated. These variables could be, for instance, RNG seed state variables or\nBatchNormalization statistics. We're going to need to return those. So we need\nsomething like this:\n\n```python\ndef compute_loss_and_updates(trainable_variables, non_trainable_variables, x, y):\n    ...\n    return loss, non_trainable_variables\n```\n\nOnce you have such a function, you can get the gradient function by\nspecifying `hax_aux` in `value_and_grad`: it tells JAX that the loss\ncomputation function returns more outputs than just the loss. Note that the loss\nshould always be the first output.\n\n```python\ngrad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)\n(loss, non_trainable_variables), grads = grad_fn(\n    trainable_variables, non_trainable_variables, x, y\n)\n```\n\nNow that we have established the basics,\nlet's implement this `compute_loss_and_updates` function.\nKeras models have a `stateless_call` method which will come in handy here.\nIt works just like `model.__call__`, but it requires you to explicitly\npass the value of all the variables in the model, and it returns not just\nthe `__call__` outputs but also the (potentially updated) non-trainable\nvariables.\n"

def compute_loss_and_updates(trainable_variables, non_trainable_variables, x, y):
    if False:
        i = 10
        return i + 15
    (y_pred, non_trainable_variables) = model.stateless_call(trainable_variables, non_trainable_variables, x)
    loss = loss_fn(y, y_pred)
    return (loss, non_trainable_variables)
"\nLet's get the gradient function:\n"
grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)
"\n### The training step function\n\nNext, let's implement the end-to-end training step, the function\nthat will both run the forward pass, compute the loss, compute the gradients,\nbut also use the optimizer to update the trainable variables. This function\nalso needs to be stateless, so it will get as input a `state` tuple that\nincludes every state element we're going to use:\n\n- `trainable_variables` and `non_trainable_variables`: the model's variables.\n- `optimizer_variables`: the optimizer's state variables,\nsuch as momentum accumulators.\n\nTo update the trainable variables, we use the optimizer's stateless method\n`stateless_apply`. It's equivalent to `optimizer.apply()`, but it requires\nalways passing `trainable_variables` and `optimizer_variables`. It returns\nboth the updated trainable variables and the updated optimizer_variables.\n"

def train_step(state, data):
    if False:
        print('Hello World!')
    (trainable_variables, non_trainable_variables, optimizer_variables) = state
    (x, y) = data
    ((loss, non_trainable_variables), grads) = grad_fn(trainable_variables, non_trainable_variables, x, y)
    (trainable_variables, optimizer_variables) = optimizer.stateless_apply(optimizer_variables, grads, trainable_variables)
    return (loss, (trainable_variables, non_trainable_variables, optimizer_variables))
"\n### Make it fast with `jax.jit`\n\nBy default, JAX operations run eagerly,\njust like in TensorFlow eager mode and PyTorch eager mode.\nAnd just like TensorFlow eager mode and PyTorch eager mode, it's pretty slow\n-- eager mode is better used as a debugging environment, not as a way to do\nany actual work. So let's make our `train_step` fast by compiling it.\n\nWhen you have a stateless JAX function, you can compile it to XLA via the \n`@jax.jit` decorator. It will get traced during its first execution, and in\nsubsequent executions you will be executing the traced graph (this is just\nlike `@tf.function(jit_compile=True)`. Let's try it:\n"

@jax.jit
def train_step(state, data):
    if False:
        while True:
            i = 10
    (trainable_variables, non_trainable_variables, optimizer_variables) = state
    (x, y) = data
    ((loss, non_trainable_variables), grads) = grad_fn(trainable_variables, non_trainable_variables, x, y)
    (trainable_variables, optimizer_variables) = optimizer.stateless_apply(optimizer_variables, grads, trainable_variables)
    return (loss, (trainable_variables, non_trainable_variables, optimizer_variables))
"\nWe're now ready to train our model. The training loop itself\nis trivial: we just repeatedly call `loss, state = train_step(state, data)`.\n\nNote:\n\n- We convert the TF tensors yielded by the `tf.data.Dataset` to NumPy\nbefore passing them to our JAX function.\n- All variables must be built beforehand:\nthe model must be built and the optimizer must be built. Since we're using a\nFunctional API model, it's already built, but if it were a subclassed model\nyou'd need to call it on a batch of data to build it.\n"
optimizer.build(model.trainable_variables)
trainable_variables = model.trainable_variables
non_trainable_variables = model.non_trainable_variables
optimizer_variables = optimizer.variables
state = (trainable_variables, non_trainable_variables, optimizer_variables)
for (step, data) in enumerate(train_dataset):
    data = (data[0].numpy(), data[1].numpy())
    (loss, state) = train_step(state, data)
    if step % 100 == 0:
        print(f'Training loss (for 1 batch) at step {step}: {float(loss):.4f}')
        print(f'Seen so far: {(step + 1) * batch_size} samples')
'\nA key thing to notice here is that the loop is entirely stateless -- the variables\nattached to the model (`model.weights`) are never getting updated during the loop.\nTheir new values are only stored in the `state` tuple. That means that at some point,\nbefore saving the model, you should be attaching the new variable values back to the model.\n\nJust call `variable.assign(new_value)` on each model variable you want to update:\n'
(trainable_variables, non_trainable_variables, optimizer_variables) = state
for (variable, value) in zip(model.trainable_variables, trainable_variables):
    variable.assign(value)
for (variable, value) in zip(model.non_trainable_variables, non_trainable_variables):
    variable.assign(value)
"\n## Low-level handling of metrics\n\nLet's add metrics monitoring to this basic training loop.\n\nYou can readily reuse built-in Keras metrics (or custom ones you wrote) in such training\nloops written from scratch. Here's the flow:\n\n- Instantiate the metric at the start of the loop\n- Include `metric_variables` in the `train_step` arguments\nand `compute_loss_and_updates` arguments.\n- Call `metric.stateless_update_state()` in the `compute_loss_and_updates` function.\nIt's equivalent to `update_state()` -- only stateless.\n- When you need to display the current value of the metric, outside the `train_step`\n(in the eager scope), attach the new metric variable values to the metric object\nand vall `metric.result()`.\n- Call `metric.reset_state()` when you need to clear the state of the metric\n(typically at the end of an epoch)\n\nLet's use this knowledge to compute `CategoricalAccuracy` on training and\nvalidation data at the end of training:\n"
model = get_model()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()

def compute_loss_and_updates(trainable_variables, non_trainable_variables, metric_variables, x, y):
    if False:
        i = 10
        return i + 15
    (y_pred, non_trainable_variables) = model.stateless_call(trainable_variables, non_trainable_variables, x)
    loss = loss_fn(y, y_pred)
    metric_variables = train_acc_metric.stateless_update_state(metric_variables, y, y_pred)
    return (loss, (non_trainable_variables, metric_variables))
grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)

@jax.jit
def train_step(state, data):
    if False:
        i = 10
        return i + 15
    (trainable_variables, non_trainable_variables, optimizer_variables, metric_variables) = state
    (x, y) = data
    ((loss, (non_trainable_variables, metric_variables)), grads) = grad_fn(trainable_variables, non_trainable_variables, metric_variables, x, y)
    (trainable_variables, optimizer_variables) = optimizer.stateless_apply(optimizer_variables, grads, trainable_variables)
    return (loss, (trainable_variables, non_trainable_variables, optimizer_variables, metric_variables))
"\nWe'll also prepare an evaluation step function:\n"

@jax.jit
def eval_step(state, data):
    if False:
        i = 10
        return i + 15
    (trainable_variables, non_trainable_variables, metric_variables) = state
    (x, y) = data
    (y_pred, non_trainable_variables) = model.stateless_call(trainable_variables, non_trainable_variables, x)
    loss = loss_fn(y, y_pred)
    metric_variables = val_acc_metric.stateless_update_state(metric_variables, y, y_pred)
    return (loss, (trainable_variables, non_trainable_variables, metric_variables))
'\nHere are our loops:\n'
optimizer.build(model.trainable_variables)
trainable_variables = model.trainable_variables
non_trainable_variables = model.non_trainable_variables
optimizer_variables = optimizer.variables
metric_variables = train_acc_metric.variables
state = (trainable_variables, non_trainable_variables, optimizer_variables, metric_variables)
for (step, data) in enumerate(train_dataset):
    data = (data[0].numpy(), data[1].numpy())
    (loss, state) = train_step(state, data)
    if step % 100 == 0:
        print(f'Training loss (for 1 batch) at step {step}: {float(loss):.4f}')
        (_, _, _, metric_variables) = state
        for (variable, value) in zip(train_acc_metric.variables, metric_variables):
            variable.assign(value)
        print(f'Training accuracy: {train_acc_metric.result()}')
        print(f'Seen so far: {(step + 1) * batch_size} samples')
metric_variables = val_acc_metric.variables
(trainable_variables, non_trainable_variables, optimizer_variables, metric_variables) = state
state = (trainable_variables, non_trainable_variables, metric_variables)
for (step, data) in enumerate(val_dataset):
    data = (data[0].numpy(), data[1].numpy())
    (loss, state) = eval_step(state, data)
    if step % 100 == 0:
        print(f'Validation loss (for 1 batch) at step {step}: {float(loss):.4f}')
        (_, _, metric_variables) = state
        for (variable, value) in zip(val_acc_metric.variables, metric_variables):
            variable.assign(value)
        print(f'Validation accuracy: {val_acc_metric.result()}')
        print(f'Seen so far: {(step + 1) * batch_size} samples')
'\n## Low-level handling of losses tracked by the model\n\nLayers & models recursively track any losses created during the forward pass\nby layers that call `self.add_loss(value)`. The resulting list of scalar loss\nvalues are available via the property `model.losses`\nat the end of the forward pass.\n\nIf you want to be using these loss components, you should sum them\nand add them to the main loss in your training step.\n\nConsider this layer, that creates an activity regularization loss:\n'

class ActivityRegularizationLayer(keras.layers.Layer):

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        self.add_loss(0.01 * jax.numpy.sum(inputs))
        return inputs
"\nLet's build a really simple model that uses it:\n"
inputs = keras.Input(shape=(784,), name='digits')
x = keras.layers.Dense(64, activation='relu')(inputs)
x = ActivityRegularizationLayer()(x)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(10, name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
"\nHere's what our `compute_loss_and_updates` function should look like now:\n\n- Pass `return_losses=True` to `model.stateless_call()`.\n- Sum the resulting `losses` and add them to the main loss.\n"

def compute_loss_and_updates(trainable_variables, non_trainable_variables, metric_variables, x, y):
    if False:
        print('Hello World!')
    (y_pred, non_trainable_variables, losses) = model.stateless_call(trainable_variables, non_trainable_variables, x, return_losses=True)
    loss = loss_fn(y, y_pred)
    if losses:
        loss += jax.numpy.sum(losses)
    metric_variables = train_acc_metric.stateless_update_state(metric_variables, y, y_pred)
    return (loss, non_trainable_variables, metric_variables)
"\nThat's it!\n"