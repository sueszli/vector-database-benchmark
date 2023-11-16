"""
Title: Writing your own callbacks
Authors: Rick Chao, Francois Chollet
Date created: 2019/03/20
Last modified: 2023/06/25
Description: Complete guide to writing new Keras callbacks.
Accelerator: GPU
"""
'\n## Introduction\n\nA callback is a powerful tool to customize the behavior of a Keras model during\ntraining, evaluation, or inference. Examples include `keras.callbacks.TensorBoard`\nto visualize training progress and results with TensorBoard, or\n`keras.callbacks.ModelCheckpoint` to periodically save your model during training.\n\nIn this guide, you will learn what a Keras callback is, what it can do, and how you can\nbuild your own. We provide a few demos of simple callback applications to get you\nstarted.\n'
'\n## Setup\n'
import numpy as np
import keras
'\n## Keras callbacks overview\n\nAll callbacks subclass the `keras.callbacks.Callback` class, and\noverride a set of methods called at various stages of training, testing, and\npredicting. Callbacks are useful to get a view on internal states and statistics of\nthe model during training.\n\nYou can pass a list of callbacks (as the keyword argument `callbacks`) to the following\nmodel methods:\n\n- `keras.Model.fit()`\n- `keras.Model.evaluate()`\n- `keras.Model.predict()`\n'
'\n## An overview of callback methods\n\n### Global methods\n\n#### `on_(train|test|predict)_begin(self, logs=None)`\n\nCalled at the beginning of `fit`/`evaluate`/`predict`.\n\n#### `on_(train|test|predict)_end(self, logs=None)`\n\nCalled at the end of `fit`/`evaluate`/`predict`.\n\n### Batch-level methods for training/testing/predicting\n\n#### `on_(train|test|predict)_batch_begin(self, batch, logs=None)`\n\nCalled right before processing a batch during training/testing/predicting.\n\n#### `on_(train|test|predict)_batch_end(self, batch, logs=None)`\n\nCalled at the end of training/testing/predicting a batch. Within this method, `logs` is\na dict containing the metrics results.\n\n### Epoch-level methods (training only)\n\n#### `on_epoch_begin(self, epoch, logs=None)`\n\nCalled at the beginning of an epoch during training.\n\n#### `on_epoch_end(self, epoch, logs=None)`\n\nCalled at the end of an epoch during training.\n'
"\n## A basic example\n\nLet's take a look at a concrete example. To get started, let's import tensorflow and\ndefine a simple Sequential Keras model:\n"

def get_model():
    if False:
        for i in range(10):
            print('nop')
    model = keras.Sequential()
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.1), loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model
'\nThen, load the MNIST data for training and testing from Keras datasets API:\n'
((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]
'\nNow, define a simple custom callback that logs:\n\n- When `fit`/`evaluate`/`predict` starts & ends\n- When each epoch starts & ends\n- When each training batch starts & ends\n- When each evaluation (test) batch starts & ends\n- When each inference (prediction) batch starts & ends\n'

class CustomCallback(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        if False:
            for i in range(10):
                print('nop')
        keys = list(logs.keys())
        print('Starting training; got log keys: {}'.format(keys))

    def on_train_end(self, logs=None):
        if False:
            while True:
                i = 10
        keys = list(logs.keys())
        print('Stop training; got log keys: {}'.format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        if False:
            while True:
                i = 10
        keys = list(logs.keys())
        print('Start epoch {} of training; got log keys: {}'.format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        if False:
            for i in range(10):
                print('nop')
        keys = list(logs.keys())
        print('End epoch {} of training; got log keys: {}'.format(epoch, keys))

    def on_test_begin(self, logs=None):
        if False:
            i = 10
            return i + 15
        keys = list(logs.keys())
        print('Start testing; got log keys: {}'.format(keys))

    def on_test_end(self, logs=None):
        if False:
            while True:
                i = 10
        keys = list(logs.keys())
        print('Stop testing; got log keys: {}'.format(keys))

    def on_predict_begin(self, logs=None):
        if False:
            while True:
                i = 10
        keys = list(logs.keys())
        print('Start predicting; got log keys: {}'.format(keys))

    def on_predict_end(self, logs=None):
        if False:
            for i in range(10):
                print('nop')
        keys = list(logs.keys())
        print('Stop predicting; got log keys: {}'.format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        if False:
            print('Hello World!')
        keys = list(logs.keys())
        print('...Training: start of batch {}; got log keys: {}'.format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        if False:
            i = 10
            return i + 15
        keys = list(logs.keys())
        print('...Training: end of batch {}; got log keys: {}'.format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        if False:
            print('Hello World!')
        keys = list(logs.keys())
        print('...Evaluating: start of batch {}; got log keys: {}'.format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        if False:
            for i in range(10):
                print('nop')
        keys = list(logs.keys())
        print('...Evaluating: end of batch {}; got log keys: {}'.format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        if False:
            i = 10
            return i + 15
        keys = list(logs.keys())
        print('...Predicting: start of batch {}; got log keys: {}'.format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        if False:
            for i in range(10):
                print('nop')
        keys = list(logs.keys())
        print('...Predicting: end of batch {}; got log keys: {}'.format(batch, keys))
"\nLet's try it out:\n"
model = get_model()
model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=0, validation_split=0.5, callbacks=[CustomCallback()])
res = model.evaluate(x_test, y_test, batch_size=128, verbose=0, callbacks=[CustomCallback()])
res = model.predict(x_test, batch_size=128, callbacks=[CustomCallback()])
'\n### Usage of `logs` dict\n\nThe `logs` dict contains the loss value, and all the metrics at the end of a batch or\nepoch. Example includes the loss and mean absolute error.\n'

class LossAndErrorPrintingCallback(keras.callbacks.Callback):

    def on_train_batch_end(self, batch, logs=None):
        if False:
            while True:
                i = 10
        print('Up to batch {}, the average loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_test_batch_end(self, batch, logs=None):
        if False:
            return 10
        print('Up to batch {}, the average loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_epoch_end(self, epoch, logs=None):
        if False:
            for i in range(10):
                print('nop')
        print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['mean_absolute_error']))
model = get_model()
model.fit(x_train, y_train, batch_size=128, epochs=2, verbose=0, callbacks=[LossAndErrorPrintingCallback()])
res = model.evaluate(x_test, y_test, batch_size=128, verbose=0, callbacks=[LossAndErrorPrintingCallback()])
"\n## Usage of `self.model` attribute\n\nIn addition to receiving log information when one of their methods is called,\ncallbacks have access to the model associated with the current round of\ntraining/evaluation/inference: `self.model`.\n\nHere are a few of the things you can do with `self.model` in a callback:\n\n- Set `self.model.stop_training = True` to immediately interrupt training.\n- Mutate hyperparameters of the optimizer (available as `self.model.optimizer`),\nsuch as `self.model.optimizer.learning_rate`.\n- Save the model at period intervals.\n- Record the output of `model.predict()` on a few test samples at the end of each\nepoch, to use as a sanity check during training.\n- Extract visualizations of intermediate features at the end of each epoch, to monitor\nwhat the model is learning over time.\n- etc.\n\nLet's see this in action in a couple of examples.\n"
'\n## Examples of Keras callback applications\n\n### Early stopping at minimum loss\n\nThis first example shows the creation of a `Callback` that stops training when the\nminimum of loss has been reached, by setting the attribute `self.model.stop_training`\n(boolean). Optionally, you can provide an argument `patience` to specify how many\nepochs we should wait before stopping after having reached a local minimum.\n\n`keras.callbacks.EarlyStopping` provides a more complete and general implementation.\n'

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, patience=0):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        if False:
            i = 10
            return i + 15
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if False:
            return 10
        current = logs.get('loss')
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if False:
            print('Hello World!')
        if self.stopped_epoch > 0:
            print(f'Epoch {self.stopped_epoch + 1}: early stopping')
model = get_model()
model.fit(x_train, y_train, batch_size=64, epochs=30, verbose=0, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
'\n### Learning rate scheduling\n\nIn this example, we show how a custom Callback can be used to dynamically change the\nlearning rate of the optimizer during the course of training.\n\nSee `callbacks.LearningRateScheduler` for a more general implementations.\n'

class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        if False:
            return 10
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if False:
            return 10
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        lr = self.model.optimizer.learning_rate
        scheduled_lr = self.schedule(epoch, lr)
        self.model.optimizer.learning_rate = scheduled_lr
        print(f'\nEpoch {epoch}: Learning rate is {float(np.array(scheduled_lr))}.')
LR_SCHEDULE = [(3, 0.05), (6, 0.01), (9, 0.005), (12, 0.001)]

def lr_schedule(epoch, lr):
    if False:
        while True:
            i = 10
    'Helper function to retrieve the scheduled learning rate based on epoch.'
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr
model = get_model()
model.fit(x_train, y_train, batch_size=64, epochs=15, verbose=0, callbacks=[LossAndErrorPrintingCallback(), CustomLearningRateScheduler(lr_schedule)])
'\n### Built-in Keras callbacks\n\nBe sure to check out the existing Keras callbacks by\nreading the [API docs](https://keras.io/api/callbacks/).\nApplications include logging to CSV, saving\nthe model, visualizing metrics in TensorBoard, and a lot more!\n'