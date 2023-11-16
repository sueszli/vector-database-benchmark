"""
Title: Writing a training loop from scratch in PyTorch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2023/06/25
Last modified: 2023/06/25
Description: Writing low-level training & evaluation loops in PyTorch.
Accelerator: None
"""
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'torch'
import torch
import keras
import numpy as np
'\n## Introduction\n\nKeras provides default training and evaluation loops, `fit()` and `evaluate()`.\nTheir usage is covered in the guide\n[Training & evaluation with the built-in methods](https://keras.io/guides/training_with_built_in_methods/).\n\nIf you want to customize the learning algorithm of your model while still leveraging\nthe convenience of `fit()`\n(for instance, to train a GAN using `fit()`), you can subclass the `Model` class and\nimplement your own `train_step()` method, which\nis called repeatedly during `fit()`.\n\nNow, if you want very low-level control over training & evaluation, you should write\nyour own training & evaluation loops from scratch. This is what this guide is about.\n'
"\n## A first end-to-end example\n\nTo write a custom training loop, we need the following ingredients:\n\n- A model to train, of course.\n- An optimizer. You could either use a `keras.optimizers` optimizer,\nor a native PyTorch optimizer from `torch.optim`.\n- A loss function. You could either use a `keras.losses` loss,\nor a native PyTorch loss from `torch.nn`.\n- A dataset. You could use any format: a `tf.data.Dataset`,\na PyTorch `DataLoader`, a Python generator, etc.\n\nLet's line them up. We'll use torch-native objects in each case --\nexcept, of course, for the Keras model.\n\nFirst, let's get the model and the MNIST dataset:\n"

def get_model():
    if False:
        return 10
    inputs = keras.Input(shape=(784,), name='digits')
    x1 = keras.layers.Dense(64, activation='relu')(inputs)
    x2 = keras.layers.Dense(64, activation='relu')(x1)
    outputs = keras.layers.Dense(10, name='predictions')(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
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
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
"\nNext, here's our PyTorch optimizer and our PyTorch loss function:\n"
model = get_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
"\nLet's train our model using mini-batch gradient with a custom training loop.\n\nCalling `loss.backward()` on a loss tensor triggers backpropagation.\nOnce that's done, your optimizer is magically aware of the gradients for each variable\nand can update its variables, which is done via `optimizer.step()`.\nTensors, variables, optimizers are all interconnected to one another via hidden global state.\nAlso, don't forget to call `model.zero_grad()` before `loss.backward()`, or you won't\nget the right gradients for your variables.\n\nHere's our training loop, step by step:\n\n- We open a `for` loop that iterates over epochs\n- For each epoch, we open a `for` loop that iterates over the dataset, in batches\n- For each batch, we call the model on the input data to retrive the predictions,\nthen we use them to compute a loss value\n- We call `loss.backward()` to \n- Outside the scope, we retrieve the gradients of the weights\nof the model with regard to the loss\n- Finally, we use the optimizer to update the weights of the model based on the\ngradients\n"
epochs = 3
for epoch in range(epochs):
    for (step, (inputs, targets)) in enumerate(train_dataloader):
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f'Training loss (for 1 batch) at step {step}: {loss.detach().numpy():.4f}')
            print(f'Seen so far: {(step + 1) * batch_size} samples')
"\nAs an alternative, let's look at what the loop looks like when using a Keras optimizer\nand a Keras loss function.\n\nImportant differences:\n\n- You retrieve the gradients for the variables via `v.value.grad`,\ncalled on each trainable variable.\n- You update your variables via `optimizer.apply()`, which must be\ncalled in a `torch.no_grad()` scope.\n\n**Also, a big gotcha:** while all NumPy/TensorFlow/JAX/Keras APIs\nas well as Python `unittest` APIs use the argument order convention\n`fn(y_true, y_pred)` (reference values first, predicted values second),\nPyTorch actually uses `fn(y_pred, y_true)` for its losses.\nSo make sure to invert the order of `logits` and `targets`.\n"
model = get_model()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
for epoch in range(epochs):
    print(f'\nStart of epoch {epoch}')
    for (step, (inputs, targets)) in enumerate(train_dataloader):
        logits = model(inputs)
        loss = loss_fn(targets, logits)
        model.zero_grad()
        trainable_weights = [v for v in model.trainable_weights]
        loss.backward()
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            optimizer.apply(gradients, trainable_weights)
        if step % 100 == 0:
            print(f'Training loss (for 1 batch) at step {step}: {loss.detach().numpy():.4f}')
            print(f'Seen so far: {(step + 1) * batch_size} samples')
"\n## Low-level handling of metrics\n\nLet's add metrics monitoring to this basic training loop.\n\nYou can readily reuse built-in Keras metrics (or custom ones you wrote) in such training\nloops written from scratch. Here's the flow:\n\n- Instantiate the metric at the start of the loop\n- Call `metric.update_state()` after each batch\n- Call `metric.result()` when you need to display the current value of the metric\n- Call `metric.reset_state()` when you need to clear the state of the metric\n(typically at the end of an epoch)\n\nLet's use this knowledge to compute `CategoricalAccuracy` on training and\nvalidation data at the end of each epoch:\n"
model = get_model()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()
"\nHere's our training & evaluation loop:\n"
for epoch in range(epochs):
    print(f'\nStart of epoch {epoch}')
    for (step, (inputs, targets)) in enumerate(train_dataloader):
        logits = model(inputs)
        loss = loss_fn(targets, logits)
        model.zero_grad()
        trainable_weights = [v for v in model.trainable_weights]
        loss.backward()
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            optimizer.apply(gradients, trainable_weights)
        train_acc_metric.update_state(targets, logits)
        if step % 100 == 0:
            print(f'Training loss (for 1 batch) at step {step}: {loss.detach().numpy():.4f}')
            print(f'Seen so far: {(step + 1) * batch_size} samples')
    train_acc = train_acc_metric.result()
    print(f'Training acc over epoch: {float(train_acc):.4f}')
    train_acc_metric.reset_state()
    for (x_batch_val, y_batch_val) in val_dataloader:
        val_logits = model(x_batch_val, training=False)
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print(f'Validation acc: {float(val_acc):.4f}')
'\n## Low-level handling of losses tracked by the model\n\nLayers & models recursively track any losses created during the forward pass\nby layers that call `self.add_loss(value)`. The resulting list of scalar loss\nvalues are available via the property `model.losses`\nat the end of the forward pass.\n\nIf you want to be using these loss components, you should sum them\nand add them to the main loss in your training step.\n\nConsider this layer, that creates an activity regularization loss:\n'

class ActivityRegularizationLayer(keras.layers.Layer):

    def call(self, inputs):
        if False:
            while True:
                i = 10
        self.add_loss(0.01 * torch.sum(inputs))
        return inputs
"\nLet's build a really simple model that uses it:\n"
inputs = keras.Input(shape=(784,), name='digits')
x = keras.layers.Dense(64, activation='relu')(inputs)
x = ActivityRegularizationLayer()(x)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(10, name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
"\nHere's what our training loop should look like now:\n"
model = get_model()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()
for epoch in range(epochs):
    print(f'\nStart of epoch {epoch}')
    for (step, (inputs, targets)) in enumerate(train_dataloader):
        logits = model(inputs)
        loss = loss_fn(targets, logits)
        if model.losses:
            loss = loss + torch.sum(*model.losses)
        model.zero_grad()
        trainable_weights = [v for v in model.trainable_weights]
        loss.backward()
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            optimizer.apply(gradients, trainable_weights)
        train_acc_metric.update_state(targets, logits)
        if step % 100 == 0:
            print(f'Training loss (for 1 batch) at step {step}: {loss.detach().numpy():.4f}')
            print(f'Seen so far: {(step + 1) * batch_size} samples')
    train_acc = train_acc_metric.result()
    print(f'Training acc over epoch: {float(train_acc):.4f}')
    train_acc_metric.reset_state()
    for (x_batch_val, y_batch_val) in val_dataloader:
        val_logits = model(x_batch_val, training=False)
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print(f'Validation acc: {float(val_acc):.4f}')
"\nThat's it!\n"