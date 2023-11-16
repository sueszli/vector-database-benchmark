"""
Title: Training & evaluation with the built-in methods
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2019/03/01
Last modified: 2023/03/25
Description: Complete guide to training & evaluation with `fit()` and `evaluate()`.
Accelerator: GPU
"""
'\n## Setup\n'
import torch
import tensorflow as tf
import os
import numpy as np
import keras
from keras import layers
from keras import ops
'\n## Introduction\n\nThis guide covers training, evaluation, and prediction (inference) models\nwhen using built-in APIs for training & validation (such as `Model.fit()`,\n`Model.evaluate()` and `Model.predict()`).\n\nIf you are interested in leveraging `fit()` while specifying your\nown training step function, see the\n[Customizing what happens in `fit()` guide](/guides/customizing_what_happens_in_fit/).\n\nIf you are interested in writing your own training & evaluation loops from\nscratch, see the guide\n["writing a training loop from scratch"](/guides/writing_a_training_loop_from_scratch/).\n\nIn general, whether you are using built-in loops or writing your own, model training &\nevaluation works strictly in the same way across every kind of Keras model --\nSequential models, models built with the Functional API, and models written from\nscratch via model subclassing.\n\nThis guide doesn\'t cover distributed training, which is covered in our\n[guide to multi-GPU & distributed training](https://keras.io/guides/distributed_training/).\n'
"\n## API overview: a first end-to-end example\n\nWhen passing data to the built-in training loops of a model, you should either use:\n\n- NumPy arrays (if your data is small and fits in memory)\n- Subclasses of `keras.utils.PyDataset`\n- `tf.data.Dataset` objects\n- PyTorch `DataLoader` instances\n\nIn the next few paragraphs, we'll use the MNIST dataset as NumPy arrays, in\norder to demonstrate how to use optimizers, losses, and metrics. Afterwards, we'll\ntake a close look at each of the other options.\n\nLet's consider the following model (here, we build in with the Functional API, but it\ncould be a Sequential model or a subclassed model as well):\n"
inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
"\nHere's what the typical end-to-end workflow looks like, consisting of:\n\n- Training\n- Validation on a holdout set generated from the original training data\n- Evaluation on the test data\n\nWe'll use MNIST data for this example.\n"
((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
'\nWe specify the training configuration (optimizer, loss, metrics):\n'
model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=[keras.metrics.SparseCategoricalAccuracy()])
'\nWe call `fit()`, which will train the model by slicing the data into "batches" of size\n`batch_size`, and repeatedly iterating over the entire dataset for a given number of\n`epochs`.\n'
print('Fit model on training data')
history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_val, y_val))
'\nThe returned `history` object holds a record of the loss values and metric values\nduring training:\n'
history.history
'\nWe evaluate the model on the test data via `evaluate()`:\n'
print('Evaluate on test data')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)
print('Generate predictions for 3 samples')
predictions = model.predict(x_test[:3])
print('predictions shape:', predictions.shape)
"\nNow, let's review each piece of this workflow in detail.\n"
'\n## The `compile()` method: specifying a loss, metrics, and an optimizer\n\nTo train a model with `fit()`, you need to specify a loss function, an optimizer, and\noptionally, some metrics to monitor.\n\nYou pass these to the model as arguments to the `compile()` method:\n'
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=[keras.metrics.SparseCategoricalAccuracy()])
"\nThe `metrics` argument should be a list -- your model can have any number of metrics.\n\nIf your model has multiple outputs, you can specify different losses and metrics for\neach output, and you can modulate the contribution of each output to the total loss of\nthe model. You will find more details about this in the **Passing data to multi-input,\nmulti-output models** section.\n\nNote that if you're satisfied with the default settings, in many cases the optimizer,\nloss, and metrics can be specified via string identifiers as a shortcut:\n"
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
"\nFor later reuse, let's put our model definition and compile step in functions; we will\ncall them several times across different examples in this guide.\n"

def get_uncompiled_model():
    if False:
        while True:
            i = 10
    inputs = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_compiled_model():
    if False:
        i = 10
        return i + 15
    model = get_uncompiled_model()
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model
"\n### Many built-in optimizers, losses, and metrics are available\n\nIn general, you won't have to create your own losses, metrics, or optimizers\nfrom scratch, because what you need is likely to be already part of the Keras API:\n\nOptimizers:\n\n- `SGD()` (with or without momentum)\n- `RMSprop()`\n- `Adam()`\n- etc.\n\nLosses:\n\n- `MeanSquaredError()`\n- `KLDivergence()`\n- `CosineSimilarity()`\n- etc.\n\nMetrics:\n\n- `AUC()`\n- `Precision()`\n- `Recall()`\n- etc.\n"
'\n### Custom losses\n\nIf you need to create a custom loss, Keras provides three ways to do so.\n\nThe first method involves creating a function that accepts inputs `y_true` and\n`y_pred`. The following example shows a loss function that computes the mean squared\nerror between the real data and the predictions:\n'

def custom_mean_squared_error(y_true, y_pred):
    if False:
        i = 10
        return i + 15
    return ops.mean(ops.square(y_true - y_pred), axis=-1)
model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(), loss=custom_mean_squared_error)
y_train_one_hot = ops.one_hot(y_train, num_classes=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)
"\nIf you need a loss function that takes in parameters beside `y_true` and `y_pred`, you\ncan subclass the `keras.losses.Loss` class and implement the following two methods:\n\n- `__init__(self)`: accept parameters to pass during the call of your loss function\n- `call(self, y_true, y_pred)`: use the targets (y_true) and the model predictions\n(y_pred) to compute the model's loss\n\nLet's say you want to use mean squared error, but with an added term that\nwill de-incentivize  prediction values far from 0.5 (we assume that the categorical\ntargets are one-hot encoded and take values between 0 and 1). This\ncreates an incentive for the model not to be too confident, which may help\nreduce overfitting (we won't know if it works until we try!).\n\nHere's how you would do it:\n"

class CustomMSE(keras.losses.Loss):

    def __init__(self, regularization_factor=0.1, name='custom_mse'):
        if False:
            while True:
                i = 10
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        if False:
            for i in range(10):
                print('nop')
        mse = ops.mean(ops.square(y_true - y_pred), axis=-1)
        reg = ops.mean(ops.square(0.5 - y_pred), axis=-1)
        return mse + reg * self.regularization_factor
model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(), loss=CustomMSE())
y_train_one_hot = ops.one_hot(y_train, num_classes=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)
"\n### Custom metrics\n\nIf you need a metric that isn't part of the API, you can easily create custom metrics\nby subclassing the `keras.metrics.Metric` class. You will need to implement 4\nmethods:\n\n- `__init__(self)`, in which you will create state variables for your metric.\n- `update_state(self, y_true, y_pred, sample_weight=None)`, which uses the targets\ny_true and the model predictions y_pred to update the state variables.\n- `result(self)`, which uses the state variables to compute the final results.\n- `reset_state(self)`, which reinitializes the state of the metric.\n\nState update and results computation are kept separate (in `update_state()` and\n`result()`, respectively) because in some cases, the results computation might be very\nexpensive and would only be done periodically.\n\nHere's a simple example showing how to implement a `CategoricalTruePositives` metric\nthat counts how many samples were correctly classified as belonging to a given class:\n"

class CategoricalTruePositives(keras.metrics.Metric):

    def __init__(self, name='categorical_true_positives', **kwargs):
        if False:
            return 10
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_variable(shape=(), name='ctp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if False:
            print('Hello World!')
        y_pred = ops.reshape(ops.argmax(y_pred, axis=1), (-1, 1))
        values = ops.cast(y_true, 'int32') == ops.cast(y_pred, 'int32')
        values = ops.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, 'float32')
            values = ops.multiply(values, sample_weight)
        self.true_positives.assign_add(ops.sum(values))

    def result(self):
        if False:
            i = 10
            return i + 15
        return self.true_positives

    def reset_state(self):
        if False:
            i = 10
            return i + 15
        self.true_positives.assign(0.0)
model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=[CategoricalTruePositives()])
model.fit(x_train, y_train, batch_size=64, epochs=3)
'\n### Handling losses and metrics that don\'t fit the standard signature\n\nThe overwhelming majority of losses and metrics can be computed from `y_true` and\n`y_pred`, where `y_pred` is an output of your model -- but not all of them. For\ninstance, a regularization loss may only require the activation of a layer (there are\nno targets in this case), and this activation may not be a model output.\n\nIn such cases, you can call `self.add_loss(loss_value)` from inside the call method of\na custom layer. Losses added in this way get added to the "main" loss during training\n(the one passed to `compile()`). Here\'s a simple example that adds activity\nregularization (note that activity regularization is built-in in all Keras layers --\nthis layer is just for the sake of providing a concrete example):\n'

class ActivityRegularizationLayer(layers.Layer):

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        self.add_loss(ops.sum(inputs) * 0.1)
        return inputs
inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = ActivityRegularizationLayer()(x)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(x_train, y_train, batch_size=64, epochs=1)
'\nNote that when you pass losses via `add_loss()`, it becomes possible to call\n`compile()` without a loss function, since the model already has a loss to minimize.\n\nConsider the following `LogisticEndpoint` layer: it takes as inputs\ntargets & logits, and it tracks a crossentropy loss via `add_loss()`.\n'

class LogisticEndpoint(keras.layers.Layer):

    def __init__(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, targets, logits, sample_weights=None):
        if False:
            while True:
                i = 10
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)
        return ops.softmax(logits)
'\nYou can use it in a model with two inputs (input data & targets), compiled without a\n`loss` argument, like this:\n'
inputs = keras.Input(shape=(3,), name='inputs')
targets = keras.Input(shape=(10,), name='targets')
logits = keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name='predictions')(targets, logits)
model = keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer='adam')
data = {'inputs': np.random.random((3, 3)), 'targets': np.random.random((3, 10))}
model.fit(data)
'\nFor more information about training multi-input models, see the section **Passing data\nto multi-input, multi-output models**.\n'
'\n### Automatically setting apart a validation holdout set\n\nIn the first end-to-end example you saw, we used the `validation_data` argument to pass\na tuple of NumPy arrays `(x_val, y_val)` to the model for evaluating a validation loss\nand validation metrics at the end of each epoch.\n\nHere\'s another option: the argument `validation_split` allows you to automatically\nreserve part of your training data for validation. The argument value represents the\nfraction of the data to be reserved for validation, so it should be set to a number\nhigher than 0 and lower than 1. For instance, `validation_split=0.2` means "use 20% of\nthe data for validation", and `validation_split=0.6` means "use 60% of the data for\nvalidation".\n\nThe way the validation is computed is by taking the last x% samples of the arrays\nreceived by the `fit()` call, before any shuffling.\n\nNote that you can only use `validation_split` when training with NumPy data.\n'
model = get_compiled_model()
model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=1)
"\n## Training & evaluation using `tf.data` Datasets\n\nIn the past few paragraphs, you've seen how to handle losses, metrics, and optimizers,\nand you've seen how to use the `validation_data` and `validation_split` arguments in\n`fit()`, when your data is passed as NumPy arrays.\n\nAnother option is to use an iterator-like, such as a `tf.data.Dataset`, a\nPyTorch `DataLoader`, or a Keras `PyDataset`. Let's take look at the former.\n\nThe `tf.data` API is a set of utilities in TensorFlow 2.0 for loading and preprocessing\ndata in a way that's fast and scalable. For a complete guide about creating `Datasets`,\nsee the [tf.data documentation](https://www.tensorflow.org/guide/data).\n\n**You can use `tf.data` to train your Keras\nmodels regardless of the backend you're using --\nwhether it's JAX, PyTorch, or TensorFlow.**\nYou can pass a `Dataset` instance directly to the methods `fit()`, `evaluate()`, and\n`predict()`:\n"
model = get_compiled_model()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(64)
model.fit(train_dataset, epochs=3)
print('Evaluate')
result = model.evaluate(test_dataset)
dict(zip(model.metrics_names, result))
'\nNote that the Dataset is reset at the end of each epoch, so it can be reused of the\nnext epoch.\n\nIf you want to run training only on a specific number of batches from this Dataset, you\ncan pass the `steps_per_epoch` argument, which specifies how many training steps the\nmodel should run using this Dataset before moving on to the next epoch.\n'
model = get_compiled_model()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
model.fit(train_dataset, epochs=3, steps_per_epoch=100)
'\nYou can also pass a `Dataset` instance as the `validation_data` argument in `fit()`:\n'
model = get_compiled_model()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)
model.fit(train_dataset, epochs=1, validation_data=val_dataset)
'\nAt the end of each epoch, the model will iterate over the validation dataset and\ncompute the validation loss and validation metrics.\n\nIf you want to run validation only on a specific number of batches from this dataset,\nyou can pass the `validation_steps` argument, which specifies how many validation\nsteps the model should run with the validation dataset before interrupting validation\nand moving on to the next epoch:\n'
model = get_compiled_model()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)
model.fit(train_dataset, epochs=1, validation_data=val_dataset, validation_steps=10)
'\nNote that the validation dataset will be reset after each use (so that you will always\nbe evaluating on the same samples from epoch to epoch).\n\nThe argument `validation_split` (generating a holdout set from the training data) is\nnot supported when training from `Dataset` objects, since this feature requires the\nability to index the samples of the datasets, which is not possible in general with\nthe `Dataset` API.\n'
"\n## Training & evaluation using `PyDataset` instances\n\n`keras.utils.PyDataset` is a utility that you can subclass to obtain\na Python generator with two important properties:\n\n- It works well with multiprocessing.\n- It can be shuffled (e.g. when passing `shuffle=True` in `fit()`).\n\nA `PyDataset` must implement two methods:\n\n- `__getitem__`\n- `__len__`\n\nThe method `__getitem__` should return a complete batch.\nIf you want to modify your dataset between epochs, you may implement `on_epoch_end`.\n\nHere's a quick example:\n"

class ExamplePyDataset(keras.utils.PyDataset):

    def __init__(self, x, y, batch_size, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        if False:
            print('Hello World!')
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return (batch_x, batch_y)
train_py_dataset = ExamplePyDataset(x_train, y_train, batch_size=32)
val_py_dataset = ExamplePyDataset(x_val, y_val, batch_size=32)
'\nTo fit the model, pass the dataset instead as the `x` argument (no need for a `y`\nargument since the dataset includes the targets), and pass the validation dataset\nas the `validation_data` argument. And no need for the `batch_size` argument, since\nthe dataset is already batched!\n'
model = get_compiled_model()
model.fit(train_py_dataset, batch_size=64, validation_data=val_py_dataset, epochs=1)
'\nEvaluating the model is just as easy:\n'
model.evaluate(val_py_dataset)
'\nImportantly, `PyDataset` objects support three common constructor arguments\nthat handle the parallel processing configuration:\n\n- `workers`: Number of workers to use in multithreading or\n    multiprocessing. Typically, you\'d set it to the number of\n    cores on your CPU.\n- `use_multiprocessing`: Whether to use Python multiprocessing for\n    parallelism. Setting this to `True` means that your\n    dataset will be replicated in multiple forked processes.\n    This is necessary to gain compute-level (rather than I/O level)\n    benefits from parallelism. However it can only be set to\n    `True` if your dataset can be safely pickled.\n- `max_queue_size`: Maximum number of batches to keep in the queue\n    when iterating over the dataset in a multithreaded or\n    multipricessed setting.\n    You can reduce this value to reduce the CPU memory consumption of\n    your dataset. It defaults to 10.\n\nBy default, multiprocessing is disabled (`use_multiprocessing=False`) and only\none thread is used. You should make sure to only turn on `use_multiprocessing` if\nyour code is running inside a Python `if __name__ == "__main__":` block in order\nto avoid issues.\n\nHere\'s a 4-thread, non-multiprocessed example:\n'
train_py_dataset = ExamplePyDataset(x_train, y_train, batch_size=32, workers=4)
val_py_dataset = ExamplePyDataset(x_val, y_val, batch_size=32, workers=4)
model = get_compiled_model()
model.fit(train_py_dataset, batch_size=64, validation_data=val_py_dataset, epochs=1)
"\n## Training & evaluation using PyTorch `DataLoader` objects\n\nAll built-in training and evaluation APIs are also compatible with `torch.utils.data.Dataset` and\n`torch.utils.data.DataLoader` objects -- regardless of whether you're using the PyTorch backend,\nor the JAX or TensorFlow backends. Let's take a look at a simple example.\n\nUnlike `PyDataset` which are batch-centric, PyTorch `Dataset` objects are sample-centric:\nthe `__len__` method returns the number of samples,\nand the `__getitem__` method returns a specific sample.\n"

class ExampleTorchDataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        if False:
            print('Hello World!')
        self.x = x
        self.y = y

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.x)

    def __getitem__(self, idx):
        if False:
            for i in range(10):
                print('nop')
        return (self.x[idx], self.y[idx])
train_torch_dataset = ExampleTorchDataset(x_train, y_train)
val_torch_dataset = ExampleTorchDataset(x_val, y_val)
'\nTo use a PyTorch Dataset, you need to wrap it into a `Dataloader` which takes care\nof batching and shuffling:\n'
train_dataloader = torch.utils.data.DataLoader(train_torch_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_torch_dataset, batch_size=32, shuffle=True)
'\nNow you can use them in the Keras API just like any other iterator:\n'
model = get_compiled_model()
model.fit(train_dataloader, batch_size=64, validation_data=val_dataloader, epochs=1)
model.evaluate(val_dataloader)
'\n## Using sample weighting and class weighting\n\nWith the default settings the weight of a sample is decided by its frequency\nin the dataset. There are two methods to weight the data, independent of\nsample frequency:\n\n* Class weights\n* Sample weights\n'
'\n### Class weights\n\nThis is set by passing a dictionary to the `class_weight` argument to\n`Model.fit()`. This dictionary maps class indices to the weight that should\nbe used for samples belonging to this class.\n\nThis can be used to balance classes without resampling, or to train a\nmodel that gives more importance to a particular class.\n\nFor instance, if class "0" is half as represented as class "1" in your data,\nyou could use `Model.fit(..., class_weight={0: 1., 1: 0.5})`.\n'
'\nHere\'s a NumPy example where we use class weights or sample weights to\ngive more importance to the correct classification of class #5 (which\nis the digit "5" in the MNIST dataset).\n'
class_weight = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 2.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0}
print('Fit with class weight')
model = get_compiled_model()
model.fit(x_train, y_train, class_weight=class_weight, batch_size=64, epochs=1)
'\n### Sample weights\n\nFor fine grained control, or if you are not building a classifier,\nyou can use "sample weights".\n\n- When training from NumPy data: Pass the `sample_weight`\n  argument to `Model.fit()`.\n- When training from `tf.data` or any other sort of iterator:\n  Yield `(input_batch, label_batch, sample_weight_batch)` tuples.\n\nA "sample weights" array is an array of numbers that specify how much weight\neach sample in a batch should have in computing the total loss. It is commonly\nused in imbalanced classification problems (the idea being to give more weight\nto rarely-seen classes).\n\nWhen the weights used are ones and zeros, the array can be used as a *mask* for\nthe loss function (entirely discarding the contribution of certain samples to\nthe total loss).\n'
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.0
print('Fit with sample weight')
model = get_compiled_model()
model.fit(x_train, y_train, sample_weight=sample_weight, batch_size=64, epochs=1)
"\nHere's a matching `Dataset` example:\n"
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.0
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, sample_weight))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
model = get_compiled_model()
model.fit(train_dataset, epochs=1)
'\n## Passing data to multi-input, multi-output models\n\nIn the previous examples, we were considering a model with a single input (a tensor of\nshape `(764,)`) and a single output (a prediction tensor of shape `(10,)`). But what\nabout models that have multiple inputs or outputs?\n\nConsider the following model, which has an image input of shape `(32, 32, 3)` (that\'s\n`(height, width, channels)`) and a time series input of shape `(None, 10)` (that\'s\n`(timesteps, features)`). Our model will have two outputs computed from the\ncombination of these inputs: a "score" (of shape `(1,)`) and a probability\ndistribution over five classes (of shape `(5,)`).\n'
image_input = keras.Input(shape=(32, 32, 3), name='img_input')
timeseries_input = keras.Input(shape=(None, 10), name='ts_input')
x1 = layers.Conv2D(3, 3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x1)
x2 = layers.Conv1D(3, 3)(timeseries_input)
x2 = layers.GlobalMaxPooling1D()(x2)
x = layers.concatenate([x1, x2])
score_output = layers.Dense(1, name='score_output')(x)
class_output = layers.Dense(5, name='class_output')(x)
model = keras.Model(inputs=[image_input, timeseries_input], outputs=[score_output, class_output])
"\nLet's plot this model, so you can clearly see what we're doing here (note that the\nshapes shown in the plot are batch shapes, rather than per-sample shapes).\n"
keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
'\nAt compilation time, we can specify different losses to different outputs, by passing\nthe loss functions as a list:\n'
model.compile(optimizer=keras.optimizers.RMSprop(0.001), loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()])
'\nIf we only passed a single loss function to the model, the same loss function would be\napplied to every output (which is not appropriate here).\n\nLikewise for metrics:\n'
model.compile(optimizer=keras.optimizers.RMSprop(0.001), loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()], metrics=[[keras.metrics.MeanAbsolutePercentageError(), keras.metrics.MeanAbsoluteError()], [keras.metrics.CategoricalAccuracy()]])
'\nSince we gave names to our output layers, we could also specify per-output losses and\nmetrics via a dict:\n'
model.compile(optimizer=keras.optimizers.RMSprop(0.001), loss={'score_output': keras.losses.MeanSquaredError(), 'class_output': keras.losses.CategoricalCrossentropy()}, metrics={'score_output': [keras.metrics.MeanAbsolutePercentageError(), keras.metrics.MeanAbsoluteError()], 'class_output': [keras.metrics.CategoricalAccuracy()]})
'\nWe recommend the use of explicit names and dicts if you have more than 2 outputs.\n\nIt\'s possible to give different weights to different output-specific losses (for\ninstance, one might wish to privilege the "score" loss in our example, by giving to 2x\nthe importance of the class loss), using the `loss_weights` argument:\n'
model.compile(optimizer=keras.optimizers.RMSprop(0.001), loss={'score_output': keras.losses.MeanSquaredError(), 'class_output': keras.losses.CategoricalCrossentropy()}, metrics={'score_output': [keras.metrics.MeanAbsolutePercentageError(), keras.metrics.MeanAbsoluteError()], 'class_output': [keras.metrics.CategoricalAccuracy()]}, loss_weights={'score_output': 2.0, 'class_output': 1.0})
'\nYou could also choose not to compute a loss for certain outputs, if these outputs are\nmeant for prediction but not for training:\n'
model.compile(optimizer=keras.optimizers.RMSprop(0.001), loss=[None, keras.losses.CategoricalCrossentropy()])
model.compile(optimizer=keras.optimizers.RMSprop(0.001), loss={'class_output': keras.losses.CategoricalCrossentropy()})
'\nPassing data to a multi-input or multi-output model in `fit()` works in a similar way as\nspecifying a loss function in compile: you can pass **lists of NumPy arrays** (with\n1:1 mapping to the outputs that received a loss function) or **dicts mapping output\nnames to NumPy arrays**.\n'
model.compile(optimizer=keras.optimizers.RMSprop(0.001), loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()])
img_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
score_targets = np.random.random_sample(size=(100, 1))
class_targets = np.random.random_sample(size=(100, 5))
model.fit([img_data, ts_data], [score_targets, class_targets], batch_size=32, epochs=1)
model.fit({'img_input': img_data, 'ts_input': ts_data}, {'score_output': score_targets, 'class_output': class_targets}, batch_size=32, epochs=1)
"\nHere's the `Dataset` use case: similarly as what we did for NumPy arrays, the `Dataset`\nshould return a tuple of dicts.\n"
train_dataset = tf.data.Dataset.from_tensor_slices(({'img_input': img_data, 'ts_input': ts_data}, {'score_output': score_targets, 'class_output': class_targets}))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
model.fit(train_dataset, epochs=1)
'\n## Using callbacks\n\nCallbacks in Keras are objects that are called at different points during training (at\nthe start of an epoch, at the end of a batch, at the end of an epoch, etc.). They\ncan be used to implement certain behaviors, such as:\n\n- Doing validation at different points during training (beyond the built-in per-epoch\nvalidation)\n- Checkpointing the model at regular intervals or when it exceeds a certain accuracy\nthreshold\n- Changing the learning rate of the model when training seems to be plateauing\n- Doing fine-tuning of the top layers when training seems to be plateauing\n- Sending email or instant message notifications when training ends or where a certain\nperformance threshold is exceeded\n- Etc.\n\nCallbacks can be passed as a list to your call to `fit()`:\n'
model = get_compiled_model()
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=1)]
model.fit(x_train, y_train, epochs=20, batch_size=64, callbacks=callbacks, validation_split=0.2)
'\n### Many built-in callbacks are available\n\nThere are many built-in callbacks already available in Keras, such as:\n\n- `ModelCheckpoint`: Periodically save the model.\n- `EarlyStopping`: Stop training when training is no longer improving the validation\nmetrics.\n- `TensorBoard`: periodically write model logs that can be visualized in\n[TensorBoard](https://www.tensorflow.org/tensorboard) (more details in the section\n"Visualization").\n- `CSVLogger`: streams loss and metrics data to a CSV file.\n- etc.\n\nSee the [callbacks documentation](/api/callbacks/) for the complete list.\n\n### Writing your own callback\n\nYou can create a custom callback by extending the base class\n`keras.callbacks.Callback`. A callback has access to its associated model through the\nclass property `self.model`.\n\nMake sure to read the\n[complete guide to writing custom callbacks](/guides/writing_your_own_callbacks/).\n\nHere\'s a simple example saving a list of per-batch loss values during training:\n'

class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs):
        if False:
            i = 10
            return i + 15
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        if False:
            print('Hello World!')
        self.per_batch_losses.append(logs.get('loss'))
"\n## Checkpointing models\n\nWhen you're training model on relatively large datasets, it's crucial to save\ncheckpoints of your model at frequent intervals.\n\nThe easiest way to achieve this is with the `ModelCheckpoint` callback:\n"
model = get_compiled_model()
callbacks = [keras.callbacks.ModelCheckpoint(filepath='mymodel_{epoch}.keras', save_best_only=True, monitor='val_loss', verbose=1)]
model.fit(x_train, y_train, epochs=2, batch_size=64, callbacks=callbacks, validation_split=0.2)
"\nThe `ModelCheckpoint` callback can be used to implement fault-tolerance:\nthe ability to restart training from the last saved state of the model in case training\ngets randomly interrupted. Here's a basic example:\n"
checkpoint_dir = './ckpt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def make_or_restore_model():
    if False:
        for i in range(10):
            print('nop')
    checkpoints = [checkpoint_dir + '/' + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print('Creating a new model')
    return get_compiled_model()
model = make_or_restore_model()
callbacks = [keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + '/model-loss={loss:.2f}.keras', save_freq=100)]
model.fit(x_train, y_train, epochs=1, callbacks=callbacks)
'\nYou call also write your own callback for saving and restoring models.\n\nFor a complete guide on serialization and saving, see the\n[guide to saving and serializing Models](/guides/serialization_and_saving/).\n'
'\n## Using learning rate schedules\n\nA common pattern when training deep learning models is to gradually reduce the learning\nas training progresses. This is generally known as "learning rate decay".\n\nThe learning decay schedule could be static (fixed in advance, as a function of the\ncurrent epoch or the current batch index), or dynamic (responding to the current\nbehavior of the model, in particular the validation loss).\n\n### Passing a schedule to an optimizer\n\nYou can easily use a static learning rate decay schedule by passing a schedule object\nas the `learning_rate` argument in your optimizer:\n'
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)
'\nSeveral built-in schedules are available: `ExponentialDecay`, `PiecewiseConstantDecay`,\n`PolynomialDecay`, and `InverseTimeDecay`.\n\n### Using callbacks to implement a dynamic learning rate schedule\n\nA dynamic learning rate schedule (for instance, decreasing the learning rate when the\nvalidation loss is no longer improving) cannot be achieved with these schedule objects,\nsince the optimizer does not have access to validation metrics.\n\nHowever, callbacks do have access to all metrics, including validation metrics! You can\nthus achieve this pattern by using a callback that modifies the current learning rate\non the optimizer. In fact, this is even built-in as the `ReduceLROnPlateau` callback.\n'
'\n## Visualizing loss and metrics during training with TensorBoard\n\nThe best way to keep an eye on your model during training is to use\n[TensorBoard](https://www.tensorflow.org/tensorboard) -- a browser-based application\nthat you can run locally that provides you with:\n\n- Live plots of the loss and metrics for training and evaluation\n- (optionally) Visualizations of the histograms of your layer activations\n- (optionally) 3D visualizations of the embedding spaces learned by your `Embedding`\nlayers\n\nIf you have installed TensorFlow with pip, you should be able to launch TensorBoard\nfrom the command line:\n\n```\ntensorboard --logdir=/full_path_to_your_logs\n```\n'
"\n### Using the TensorBoard callback\n\nThe easiest way to use TensorBoard with a Keras model and the `fit()` method is the\n`TensorBoard` callback.\n\nIn the simplest case, just specify where you want the callback to write logs, and\nyou're good to go:\n"
keras.callbacks.TensorBoard(log_dir='/full_path_to_your_logs', histogram_freq=0, embeddings_freq=0, update_freq='epoch')
'\nFor more information, see the\n[documentation for the `TensorBoard` callback](https://keras.io/api/callbacks/tensorboard/).\n'