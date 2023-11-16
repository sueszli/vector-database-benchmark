"""
Title: Trainer pattern
Author: [nkovela1](https://nkovela1.github.io/)
Date created: 2022/09/19
Last modified: 2022/09/26
Description: Guide on how to share a custom training step across multiple Keras models.
Accelerator: GPU
"""
'\n## Introduction\n\nThis example shows how to create a custom training step using the "Trainer pattern",\nwhich can then be shared across multiple Keras models. This pattern overrides the\n`train_step()` method of the `keras.Model` class, allowing for training loops\nbeyond plain supervised learning.\n\nThe Trainer pattern can also easily be adapted to more complex models with larger\ncustom training steps, such as\n[this end-to-end GAN model](https://keras.io/guides/customizing_what_happens_in_fit/#wrapping-up-an-endtoend-gan-example),\nby putting the custom training step in the Trainer class definition.\n'
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import keras
mnist = keras.datasets.mnist
((x_train, y_train), (x_test, y_test)) = mnist.load_data()
(x_train, x_test) = (x_train / 255.0, x_test / 255.0)
'\n## Define the Trainer class\n\nA custom training and evaluation step can be created by overriding\nthe `train_step()` and `test_step()` method of a `Model` subclass:\n'

class MyTrainer(keras.Model):

    def __init__(self, model):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.model = model
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        self.accuracy_metric = keras.metrics.SparseCategoricalAccuracy()

    @property
    def metrics(self):
        if False:
            return 10
        return [self.accuracy_metric]

    def train_step(self, data):
        if False:
            return 10
        (x, y) = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y, y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        for metric in self.metrics:
            metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = data
        y_pred = self.model(x, training=False)
        for metric in self.metrics:
            metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.model(x)
        return x
"\n## Define multiple models to share the custom training step\n\nLet's define two different models that can share our Trainer class and its custom `train_step()`:\n"
model_a = keras.models.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(256, activation='relu'), keras.layers.Dropout(0.2), keras.layers.Dense(10, activation='softmax')])
func_input = keras.Input(shape=(28, 28, 1))
x = keras.layers.Flatten(input_shape=(28, 28))(func_input)
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.Dropout(0.4)(x)
func_output = keras.layers.Dense(10, activation='softmax')(x)
model_b = keras.Model(func_input, func_output)
'\n## Create Trainer class objects from the models\n'
trainer_1 = MyTrainer(model_a)
trainer_2 = MyTrainer(model_b)
'\n## Compile and fit the models to the MNIST dataset\n'
trainer_1.compile(optimizer=keras.optimizers.SGD())
trainer_1.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
trainer_2.compile(optimizer=keras.optimizers.Adam())
trainer_2.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))