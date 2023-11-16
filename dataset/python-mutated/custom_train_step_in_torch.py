"""
Title: Customizing what happens in `fit()` with PyTorch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2023/06/27
Last modified: 2023/06/27
Description: Overriding the training step of the Model class with PyTorch.
Accelerator: GPU
"""
"\n## Introduction\n\nWhen you're doing supervised learning, you can use `fit()` and everything works\nsmoothly.\n\nWhen you need to take control of every little detail, you can write your own training\nloop entirely from scratch.\n\nBut what if you need a custom training algorithm, but you still want to benefit from\nthe convenient features of `fit()`, such as callbacks, built-in distribution support,\nor step fusing?\n\nA core principle of Keras is **progressive disclosure of complexity**. You should\nalways be able to get into lower-level workflows in a gradual way. You shouldn't fall\noff a cliff if the high-level functionality doesn't exactly match your use case. You\nshould be able to gain more control over the small details while retaining a\ncommensurate amount of high-level convenience.\n\nWhen you need to customize what `fit()` does, you should **override the training step\nfunction of the `Model` class**. This is the function that is called by `fit()` for\nevery batch of data. You will then be able to call `fit()` as usual -- and it will be\nrunning your own learning algorithm.\n\nNote that this pattern does not prevent you from building models with the Functional\nAPI. You can do this whether you're building `Sequential` models, Functional API\nmodels, or subclassed models.\n\nLet's see how that works.\n"
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'torch'
import torch
import keras
from keras import layers
import numpy as np
"\n## A first simple example\n\nLet's start from a simple example:\n\n- We create a new class that subclasses `keras.Model`.\n- We just override the method `train_step(self, data)`.\n- We return a dictionary mapping metric names (including the loss) to their current\nvalue.\n\nThe input argument `data` is what gets passed to fit as training data:\n\n- If you pass NumPy arrays, by calling `fit(x, y, ...)`, then `data` will be the tuple\n`(x, y)`\n- If you pass a `torch.utils.data.DataLoader` or a `tf.data.Dataset`,\nby calling `fit(dataset, ...)`, then `data` will be what gets yielded\nby `dataset` at each batch.\n\nIn the body of the `train_step()` method, we implement a regular training update,\nsimilar to what you are already familiar with. Importantly, **we compute the loss via\n`self.compute_loss()`**, which wraps the loss(es) function(s) that were passed to\n`compile()`.\n\nSimilarly, we call `metric.update_state(y, y_pred)` on metrics from `self.metrics`,\nto update the state of the metrics that were passed in `compile()`,\nand we query results from `self.metrics` at the end to retrieve their current value.\n"

class CustomModel(keras.Model):

    def train_step(self, data):
        if False:
            i = 10
            return i + 15
        (x, y) = data
        self.zero_grad()
        y_pred = self(x, training=True)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
"\nLet's try this out:\n"
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=3)
"\n## Going lower-level\n\nNaturally, you could just skip passing a loss function in `compile()`, and instead do\neverything *manually* in `train_step`. Likewise for metrics.\n\nHere's a lower-level example, that only uses `compile()` to configure the optimizer:\n\n- We start by creating `Metric` instances to track our loss and a MAE score (in `__init__()`).\n- We implement a custom `train_step()` that updates the state of these metrics\n(by calling `update_state()` on them), then query them (via `result()`) to return their current average value,\nto be displayed by the progress bar and to be pass to any callback.\n- Note that we would need to call `reset_states()` on our metrics between each epoch! Otherwise\ncalling `result()` would return an average since the start of training, whereas we usually work\nwith per-epoch averages. Thankfully, the framework can do that for us: just list any metric\nyou want to reset in the `metrics` property of the model. The model will call `reset_states()`\non any object listed here at the beginning of each `fit()` epoch or at the beginning of a call to\n`evaluate()`.\n"

class CustomModel(keras.Model):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.mae_metric = keras.metrics.MeanAbsoluteError(name='mae')
        self.loss_fn = keras.losses.MeanSquaredError()

    def train_step(self, data):
        if False:
            while True:
                i = 10
        (x, y) = data
        self.zero_grad()
        y_pred = self(x, training=True)
        loss = self.loss_fn(y, y_pred)
        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        return {'loss': self.loss_tracker.result(), 'mae': self.mae_metric.result()}

    @property
    def metrics(self):
        if False:
            i = 10
            return i + 15
        return [self.loss_tracker, self.mae_metric]
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam')
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=5)
"\n## Supporting `sample_weight` & `class_weight`\n\nYou may have noticed that our first basic example didn't make any mention of sample\nweighting. If you want to support the `fit()` arguments `sample_weight` and\n`class_weight`, you'd simply do the following:\n\n- Unpack `sample_weight` from the `data` argument\n- Pass it to `compute_loss` & `update_state` (of course, you could also just apply\nit manually if you don't rely on `compile()` for losses & metrics)\n- That's it.\n"

class CustomModel(keras.Model):

    def train_step(self, data):
        if False:
            while True:
                i = 10
        if len(data) == 3:
            (x, y, sample_weight) = data
        else:
            sample_weight = None
            (x, y) = data
        self.zero_grad()
        y_pred = self(x, training=True)
        loss = self.compute_loss(y=y, y_pred=y_pred, sample_weight=sample_weight)
        loss.backward()
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
sw = np.random.random((1000, 1))
model.fit(x, y, sample_weight=sw, epochs=3)
"\n## Providing your own evaluation step\n\nWhat if you want to do the same for calls to `model.evaluate()`? Then you would\noverride `test_step` in exactly the same way. Here's what it looks like:\n"

class CustomModel(keras.Model):

    def test_step(self, data):
        if False:
            i = 10
            return i + 15
        (x, y) = data
        y_pred = self(x, training=False)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(loss='mse', metrics=['mae'])
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.evaluate(x, y)
'\n## Wrapping up: an end-to-end GAN example\n\nLet\'s walk through an end-to-end example that leverages everything you just learned.\n\nLet\'s consider:\n\n- A generator network meant to generate 28x28x1 images.\n- A discriminator network meant to classify 28x28x1 images into two classes ("fake" and\n"real").\n- One optimizer for each.\n- A loss function to train the discriminator.\n'
discriminator = keras.Sequential([keras.Input(shape=(28, 28, 1)), layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'), layers.LeakyReLU(negative_slope=0.2), layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'), layers.LeakyReLU(negative_slope=0.2), layers.GlobalMaxPooling2D(), layers.Dense(1)], name='discriminator')
latent_dim = 128
generator = keras.Sequential([keras.Input(shape=(latent_dim,)), layers.Dense(7 * 7 * 128), layers.LeakyReLU(negative_slope=0.2), layers.Reshape((7, 7, 128)), layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'), layers.LeakyReLU(negative_slope=0.2), layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'), layers.LeakyReLU(negative_slope=0.2), layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')], name='generator')
"\nHere's a feature-complete GAN class, overriding `compile()` to use its own signature,\nand implementing the entire GAN algorithm in 17 lines in `train_step`:\n"

class GAN(keras.Model):

    def __init__(self, discriminator, generator, latent_dim):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
        self.seed_generator = keras.random.SeedGenerator(1337)
        self.built = True

    @property
    def metrics(self):
        if False:
            return 10
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        if False:
            for i in range(10):
                print('nop')
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if False:
            return 10
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = real_images.shape[0]
        random_latent_vectors = keras.random.normal(shape=(batch_size, self.latent_dim), seed=self.seed_generator)
        generated_images = self.generator(random_latent_vectors)
        real_images = torch.tensor(real_images)
        combined_images = torch.concat([generated_images, real_images], axis=0)
        labels = torch.concat([torch.ones((batch_size, 1)), torch.zeros((batch_size, 1))], axis=0)
        labels += 0.05 * keras.random.uniform(labels.shape, seed=self.seed_generator)
        self.zero_grad()
        predictions = self.discriminator(combined_images)
        d_loss = self.loss_fn(labels, predictions)
        d_loss.backward()
        grads = [v.value.grad for v in self.discriminator.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.discriminator.trainable_weights)
        random_latent_vectors = keras.random.normal(shape=(batch_size, self.latent_dim), seed=self.seed_generator)
        misleading_labels = torch.zeros((batch_size, 1))
        self.zero_grad()
        predictions = self.discriminator(self.generator(random_latent_vectors))
        g_loss = self.loss_fn(misleading_labels, predictions)
        grads = g_loss.backward()
        grads = [v.value.grad for v in self.generator.trainable_weights]
        with torch.no_grad():
            self.g_optimizer.apply(grads, self.generator.trainable_weights)
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {'d_loss': self.d_loss_tracker.result(), 'g_loss': self.g_loss_tracker.result()}
"\nLet's test-drive it:\n"
batch_size = 64
((x_train, _), (x_test, _)) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype('float32') / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = torch.utils.data.TensorDataset(torch.from_numpy(all_digits), torch.from_numpy(all_digits))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0003), g_optimizer=keras.optimizers.Adam(learning_rate=0.0003), loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))
gan.fit(dataloader, epochs=1)
'\nThe ideas behind deep learning are simple, so why should their implementation be painful?\n'