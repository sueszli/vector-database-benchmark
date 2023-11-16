"""
Title: Customizing what happens in `fit()` with JAX
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2023/06/27
Last modified: 2023/06/27
Description: Overriding the training step of the Model class with JAX.
Accelerator: GPU
"""
"\n## Introduction\n\nWhen you're doing supervised learning, you can use `fit()` and everything works\nsmoothly.\n\nWhen you need to take control of every little detail, you can write your own training\nloop entirely from scratch.\n\nBut what if you need a custom training algorithm, but you still want to benefit from\nthe convenient features of `fit()`, such as callbacks, built-in distribution support,\nor step fusing?\n\nA core principle of Keras is **progressive disclosure of complexity**. You should\nalways be able to get into lower-level workflows in a gradual way. You shouldn't fall\noff a cliff if the high-level functionality doesn't exactly match your use case. You\nshould be able to gain more control over the small details while retaining a\ncommensurate amount of high-level convenience.\n\nWhen you need to customize what `fit()` does, you should **override the training step\nfunction of the `Model` class**. This is the function that is called by `fit()` for\nevery batch of data. You will then be able to call `fit()` as usual -- and it will be\nrunning your own learning algorithm.\n\nNote that this pattern does not prevent you from building models with the Functional\nAPI. You can do this whether you're building `Sequential` models, Functional API\nmodels, or subclassed models.\n\nLet's see how that works.\n"
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'jax'
import jax
import keras
import numpy as np
"\n## A first simple example\n\nLet's start from a simple example:\n\n- We create a new class that subclasses `keras.Model`.\n- We implement a fully-stateless `compute_loss_and_updates()` method\nto compute the loss as well as the updated values for the non-trainable\nvariables of the model. Internally, it calls `stateless_call()` and\nthe built-in `compute_loss()`.\n- We implement a fully-stateless `train_step()` method to compute current\nmetric values (including the loss) as well as updated values for the \ntrainable variables, the optimizer variables, and the metric variables.\n\nNote that you can also take into account the `sample_weight` argument by:\n\n- Unpacking the data as `x, y, sample_weight = data`\n- Passing `sample_weight` to `compute_loss()`\n- Passing `sample_weight` alongside `y` and `y_pred`\nto metrics in `stateless_update_state()`\n"

class CustomModel(keras.Model):

    def compute_loss_and_updates(self, trainable_variables, non_trainable_variables, x, y, training=False):
        if False:
            print('Hello World!')
        (y_pred, non_trainable_variables) = self.stateless_call(trainable_variables, non_trainable_variables, x, training=training)
        loss = self.compute_loss(x, y, y_pred)
        return (loss, (y_pred, non_trainable_variables))

    def train_step(self, state, data):
        if False:
            print('Hello World!')
        (trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables) = state
        (x, y) = data
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)
        ((loss, (y_pred, non_trainable_variables)), grads) = grad_fn(trainable_variables, non_trainable_variables, x, y, training=True)
        (trainable_variables, optimizer_variables) = self.optimizer.stateless_apply(optimizer_variables, grads, trainable_variables)
        new_metrics_vars = []
        for metric in self.metrics:
            this_metric_vars = metrics_variables[len(new_metrics_vars):len(new_metrics_vars) + len(metric.variables)]
            if metric.name == 'loss':
                this_metric_vars = metric.stateless_update_state(this_metric_vars, loss)
            else:
                this_metric_vars = metric.stateless_update_state(this_metric_vars, y, y_pred)
            logs = metric.stateless_result(this_metric_vars)
            new_metrics_vars += this_metric_vars
        state = (trainable_variables, non_trainable_variables, optimizer_variables, new_metrics_vars)
        return (logs, state)
"\nLet's try this out:\n"
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=3)
"\n## Going lower-level\n\nNaturally, you could just skip passing a loss function in `compile()`, and instead do\neverything *manually* in `train_step`. Likewise for metrics.\n\nHere's a lower-level example, that only uses `compile()` to configure the optimizer:\n"

class CustomModel(keras.Model):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.mae_metric = keras.metrics.MeanAbsoluteError(name='mae')
        self.loss_fn = keras.losses.MeanSquaredError()

    def compute_loss_and_updates(self, trainable_variables, non_trainable_variables, x, y, training=False):
        if False:
            print('Hello World!')
        (y_pred, non_trainable_variables) = self.stateless_call(trainable_variables, non_trainable_variables, x, training=training)
        loss = self.loss_fn(y, y_pred)
        return (loss, (y_pred, non_trainable_variables))

    def train_step(self, state, data):
        if False:
            i = 10
            return i + 15
        (trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables) = state
        (x, y) = data
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)
        ((loss, (y_pred, non_trainable_variables)), grads) = grad_fn(trainable_variables, non_trainable_variables, x, y, training=True)
        (trainable_variables, optimizer_variables) = self.optimizer.stateless_apply(optimizer_variables, grads, trainable_variables)
        loss_tracker_vars = metrics_variables[:len(self.loss_tracker.variables)]
        mae_metric_vars = metrics_variables[len(self.loss_tracker.variables):]
        loss_tracker_vars = self.loss_tracker.stateless_update_state(loss_tracker_vars, loss)
        mae_metric_vars = self.mae_metric.stateless_update_state(mae_metric_vars, y, y_pred)
        logs = {}
        logs[self.loss_tracker.name] = self.loss_tracker.stateless_result(loss_tracker_vars)
        logs[self.mae_metric.name] = self.mae_metric.stateless_result(mae_metric_vars)
        new_metrics_vars = loss_tracker_vars + mae_metric_vars
        state = (trainable_variables, non_trainable_variables, optimizer_variables, new_metrics_vars)
        return (logs, state)

    @property
    def metrics(self):
        if False:
            print('Hello World!')
        return [self.loss_tracker, self.mae_metric]
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam')
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=5)
"\n## Providing your own evaluation step\n\nWhat if you want to do the same for calls to `model.evaluate()`? Then you would\noverride `test_step` in exactly the same way. Here's what it looks like:\n"

class CustomModel(keras.Model):

    def test_step(self, state, data):
        if False:
            return 10
        (x, y) = data
        (trainable_variables, non_trainable_variables, metrics_variables) = state
        (y_pred, non_trainable_variables) = self.stateless_call(trainable_variables, non_trainable_variables, x, training=False)
        loss = self.compute_loss(x, y, y_pred)
        new_metrics_vars = []
        for metric in self.metrics:
            this_metric_vars = metrics_variables[len(new_metrics_vars):len(new_metrics_vars) + len(metric.variables)]
            if metric.name == 'loss':
                this_metric_vars = metric.stateless_update_state(this_metric_vars, loss)
            else:
                this_metric_vars = metric.stateless_update_state(this_metric_vars, y, y_pred)
            logs = metric.stateless_result(this_metric_vars)
            new_metrics_vars += this_metric_vars
        state = (trainable_variables, non_trainable_variables, new_metrics_vars)
        return (logs, state)
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(loss='mse', metrics=['mae'])
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.evaluate(x, y)
"\nThat's it!\n"