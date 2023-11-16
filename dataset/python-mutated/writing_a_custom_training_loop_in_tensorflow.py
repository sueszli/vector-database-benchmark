"""
Title: Writing a training loop from scratch in TensorFlow
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2019/03/01
Last modified: 2023/06/25
Description: Writing low-level training & evaluation loops in TensorFlow.
Accelerator: None
"""
'\n## Setup\n'
import time
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import keras
import numpy as np
'\n## Introduction\n\nKeras provides default training and evaluation loops, `fit()` and `evaluate()`.\nTheir usage is covered in the guide\n[Training & evaluation with the built-in methods](https://keras.io/guides/training_with_built_in_methods/).\n\nIf you want to customize the learning algorithm of your model while still leveraging\nthe convenience of `fit()`\n(for instance, to train a GAN using `fit()`), you can subclass the `Model` class and\nimplement your own `train_step()` method, which\nis called repeatedly during `fit()`.\n\nNow, if you want very low-level control over training & evaluation, you should write\nyour own training & evaluation loops from scratch. This is what this guide is about.\n'
"\n## A first end-to-end example\n\nLet's consider a simple MNIST model:\n"

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
"\nLet's train it using mini-batch gradient with a custom training loop.\n\nFirst, we're going to need an optimizer, a loss function, and a dataset:\n"
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
batch_size = 32
((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)
"\nCalling a model inside a `GradientTape` scope enables you to retrieve the gradients of\nthe trainable weights of the layer with respect to a loss value. Using an optimizer\ninstance, you can use these gradients to update these variables (which you can\nretrieve using `model.trainable_weights`).\n\nHere's our training loop, step by step:\n\n- We open a `for` loop that iterates over epochs\n- For each epoch, we open a `for` loop that iterates over the dataset, in batches\n- For each batch, we open a `GradientTape()` scope\n- Inside this scope, we call the model (forward pass) and compute the loss\n- Outside the scope, we retrieve the gradients of the weights\nof the model with regard to the loss\n- Finally, we use the optimizer to update the weights of the model based on the\ngradients\n"
epochs = 3
for epoch in range(epochs):
    print(f'\nStart of epoch {epoch}')
    for (step, (x_batch_train, y_batch_train)) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply(grads, model.trainable_weights)
        if step % 100 == 0:
            print(f'Training loss (for 1 batch) at step {step}: {float(loss_value):.4f}')
            print(f'Seen so far: {(step + 1) * batch_size} samples')
"\n## Low-level handling of metrics\n\nLet's add metrics monitoring to this basic loop.\n\nYou can readily reuse the built-in metrics (or custom ones you wrote) in such training\nloops written from scratch. Here's the flow:\n\n- Instantiate the metric at the start of the loop\n- Call `metric.update_state()` after each batch\n- Call `metric.result()` when you need to display the current value of the metric\n- Call `metric.reset_state()` when you need to clear the state of the metric\n(typically at the end of an epoch)\n\nLet's use this knowledge to compute `SparseCategoricalAccuracy` on training and\nvalidation data at the end of each epoch:\n"
model = get_model()
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
"\nHere's our training & evaluation loop:\n"
epochs = 2
for epoch in range(epochs):
    print(f'\nStart of epoch {epoch}')
    start_time = time.time()
    for (step, (x_batch_train, y_batch_train)) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply(grads, model.trainable_weights)
        train_acc_metric.update_state(y_batch_train, logits)
        if step % 100 == 0:
            print(f'Training loss (for 1 batch) at step {step}: {float(loss_value):.4f}')
            print(f'Seen so far: {(step + 1) * batch_size} samples')
    train_acc = train_acc_metric.result()
    print(f'Training acc over epoch: {float(train_acc):.4f}')
    train_acc_metric.reset_state()
    for (x_batch_val, y_batch_val) in val_dataset:
        val_logits = model(x_batch_val, training=False)
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print(f'Validation acc: {float(val_acc):.4f}')
    print(f'Time taken: {time.time() - start_time:.2f}s')
'\n## Speeding-up your training step with `tf.function`\n\nThe default runtime in TensorFlow is eager execution.\nAs such, our training loop above executes eagerly.\n\nThis is great for debugging, but graph compilation has a definite performance\nadvantage. Describing your computation as a static graph enables the framework\nto apply global performance optimizations. This is impossible when\nthe framework is constrained to greedily execute one operation after another,\nwith no knowledge of what comes next.\n\nYou can compile into a static graph any function that takes tensors as input.\nJust add a `@tf.function` decorator on it, like this:\n'

@tf.function
def train_step(x, y):
    if False:
        print('Hello World!')
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply(grads, model.trainable_weights)
    train_acc_metric.update_state(y, logits)
    return loss_value
"\nLet's do the same with the evaluation step:\n"

@tf.function
def test_step(x, y):
    if False:
        return 10
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)
"\nNow, let's re-run our training loop with this compiled training step:\n"
epochs = 2
for epoch in range(epochs):
    print(f'\nStart of epoch {epoch}')
    start_time = time.time()
    for (step, (x_batch_train, y_batch_train)) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)
        if step % 100 == 0:
            print(f'Training loss (for 1 batch) at step {step}: {float(loss_value):.4f}')
            print(f'Seen so far: {(step + 1) * batch_size} samples')
    train_acc = train_acc_metric.result()
    print(f'Training acc over epoch: {float(train_acc):.4f}')
    train_acc_metric.reset_state()
    for (x_batch_val, y_batch_val) in val_dataset:
        test_step(x_batch_val, y_batch_val)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print(f'Validation acc: {float(val_acc):.4f}')
    print(f'Time taken: {time.time() - start_time:.2f}s')
"\nMuch faster, isn't it?\n"
'\n## Low-level handling of losses tracked by the model\n\nLayers & models recursively track any losses created during the forward pass\nby layers that call `self.add_loss(value)`. The resulting list of scalar loss\nvalues are available via the property `model.losses`\nat the end of the forward pass.\n\nIf you want to be using these loss components, you should sum them\nand add them to the main loss in your training step.\n\nConsider this layer, that creates an activity regularization loss:\n\n'

class ActivityRegularizationLayer(keras.layers.Layer):

    def call(self, inputs):
        if False:
            return 10
        self.add_loss(0.01 * tf.reduce_sum(inputs))
        return inputs
"\nLet's build a really simple model that uses it:\n"
inputs = keras.Input(shape=(784,), name='digits')
x = keras.layers.Dense(64, activation='relu')(inputs)
x = ActivityRegularizationLayer()(x)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(10, name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
"\nHere's what our training step should look like now:\n"

@tf.function
def train_step(x, y):
    if False:
        while True:
            i = 10
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply(grads, model.trainable_weights)
    train_acc_metric.update_state(y, logits)
    return loss_value
"\n## Summary\n\nNow you know everything there is to know about using built-in training loops and\nwriting your own from scratch.\n\nTo conclude, here's a simple end-to-end example that ties together everything\nyou've learned in this guide: a DCGAN trained on MNIST digits.\n"
'\n## End-to-end example: a GAN training loop from scratch\n\nYou may be familiar with Generative Adversarial Networks (GANs). GANs can generate new\nimages that look almost real, by learning the latent distribution of a training\ndataset of images (the "latent space" of the images).\n\nA GAN is made of two parts: a "generator" model that maps points in the latent\nspace to points in image space, a "discriminator" model, a classifier\nthat can tell the difference between real images (from the training dataset)\nand fake images (the output of the generator network).\n\nA GAN training loop looks like this:\n\n1) Train the discriminator.\n- Sample a batch of random points in the latent space.\n- Turn the points into fake images via the "generator" model.\n- Get a batch of real images and combine them with the generated images.\n- Train the "discriminator" model to classify generated vs. real images.\n\n2) Train the generator.\n- Sample random points in the latent space.\n- Turn the points into fake images via the "generator" network.\n- Get a batch of real images and combine them with the generated images.\n- Train the "generator" model to "fool" the discriminator and classify the fake images\nas real.\n\nFor a much more detailed overview of how GANs works, see\n[Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python).\n\nLet\'s implement this training loop. First, create the discriminator meant to classify\nfake vs real digits:\n'
discriminator = keras.Sequential([keras.Input(shape=(28, 28, 1)), keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'), keras.layers.LeakyReLU(negative_slope=0.2), keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'), keras.layers.LeakyReLU(negative_slope=0.2), keras.layers.GlobalMaxPooling2D(), keras.layers.Dense(1)], name='discriminator')
discriminator.summary()
"\nThen let's create a generator network,\nthat turns latent vectors into outputs of shape `(28, 28, 1)` (representing\nMNIST digits):\n"
latent_dim = 128
generator = keras.Sequential([keras.Input(shape=(latent_dim,)), keras.layers.Dense(7 * 7 * 128), keras.layers.LeakyReLU(negative_slope=0.2), keras.layers.Reshape((7, 7, 128)), keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'), keras.layers.LeakyReLU(negative_slope=0.2), keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'), keras.layers.LeakyReLU(negative_slope=0.2), keras.layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')], name='generator')
"\nHere's the key bit: the training loop. As you can see it is quite straightforward. The\ntraining step function only takes 17 lines.\n"
d_optimizer = keras.optimizers.Adam(learning_rate=0.0003)
g_optimizer = keras.optimizers.Adam(learning_rate=0.0004)
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

@tf.function
def train_step(real_images):
    if False:
        i = 10
        return i + 15
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    generated_images = generator(random_latent_vectors)
    combined_images = tf.concat([generated_images, real_images], axis=0)
    labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((real_images.shape[0], 1))], axis=0)
    labels += 0.05 * tf.random.uniform(labels.shape)
    with tf.GradientTape() as tape:
        predictions = discriminator(combined_images)
        d_loss = loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply(grads, discriminator.trainable_weights)
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    misleading_labels = tf.zeros((batch_size, 1))
    with tf.GradientTape() as tape:
        predictions = discriminator(generator(random_latent_vectors))
        g_loss = loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply(grads, generator.trainable_weights)
    return (d_loss, g_loss, generated_images)
"\nLet's train our GAN, by repeatedly calling `train_step` on batches of images.\n\nSince our discriminator and generator are convnets, you're going to want to\nrun this code on a GPU.\n"
batch_size = 64
((x_train, _), (x_test, _)) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype('float32') / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
epochs = 1
save_dir = './'
for epoch in range(epochs):
    print(f'\nStart epoch {epoch}')
    for (step, real_images) in enumerate(dataset):
        (d_loss, g_loss, generated_images) = train_step(real_images)
        if step % 100 == 0:
            print(f'discriminator loss at step {step}: {d_loss:.2f}')
            print(f'adversarial loss at step {step}: {g_loss:.2f}')
            img = keras.utils.array_to_img(generated_images[0] * 255.0, scale=False)
            img.save(os.path.join(save_dir, f'generated_img_{step}.png'))
        if step > 10:
            break
"\nThat's it! You'll get nice-looking fake MNIST digits after just ~30s of training on the\nColab GPU.\n"