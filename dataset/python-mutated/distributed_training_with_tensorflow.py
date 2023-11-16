"""
Title: Multi-GPU distributed training with TensorFlow
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/28
Last modified: 2023/06/29
Description: Guide to multi-GPU training for Keras models with TensorFlow.
Accelerator: GPU
"""
'\n## Introduction\n\nThere are generally two ways to distribute computation across multiple devices:\n\n**Data parallelism**, where a single model gets replicated on multiple devices or\nmultiple machines. Each of them processes different batches of data, then they merge\ntheir results. There exist many variants of this setup, that differ in how the different\nmodel replicas merge results, in whether they stay in sync at every batch or whether they\nare more loosely coupled, etc.\n\n**Model parallelism**, where different parts of a single model run on different devices,\nprocessing a single batch of data together. This works best with models that have a\nnaturally-parallel architecture, such as models that feature multiple branches.\n\nThis guide focuses on data parallelism, in particular **synchronous data parallelism**,\nwhere the different replicas of the model stay in sync after each batch they process.\nSynchronicity keeps the model convergence behavior identical to what you would see for\nsingle-device training.\n\nSpecifically, this guide teaches you how to use the `tf.distribute` API to train Keras\nmodels on multiple GPUs, with minimal changes to your code,\non multiple GPUs (typically 2 to 16) installed on a single machine (single host,\nmulti-device training). This is the most common setup for researchers and small-scale\nindustry workflows.\n'
'\n## Setup\n'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import keras
"\n## Single-host, multi-device synchronous training\n\nIn this setup, you have one machine with several GPUs on it (typically 2 to 16). Each\ndevice will run a copy of your model (called a **replica**). For simplicity, in what\nfollows, we'll assume we're dealing with 8 GPUs, at no loss of generality.\n\n**How it works**\n\nAt each step of training:\n\n- The current batch of data (called **global batch**) is split into 8 different\nsub-batches (called **local batches**). For instance, if the global batch has 512\nsamples, each of the 8 local batches will have 64 samples.\n- Each of the 8 replicas independently processes a local batch: they run a forward pass,\nthen a backward pass, outputting the gradient of the weights with respect to the loss of\nthe model on the local batch.\n- The weight updates originating from local gradients are efficiently merged across the 8\nreplicas. Because this is done at the end of every step, the replicas always stay in\nsync.\n\nIn practice, the process of synchronously updating the weights of the model replicas is\nhandled at the level of each individual weight variable. This is done through a **mirrored\nvariable** object.\n\n**How to use it**\n\nTo do single-host, multi-device synchronous training with a Keras model, you would use\nthe [`tf.distribute.MirroredStrategy` API](\n    https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy).\nHere's how it works:\n\n- Instantiate a `MirroredStrategy`, optionally configuring which specific devices you\nwant to use (by default the strategy will use all GPUs available).\n- Use the strategy object to open a scope, and within this scope, create all the Keras\nobjects you need that contain variables. Typically, that means **creating & compiling the\nmodel** inside the distribution scope. In some cases, the first call to `fit()` may also\ncreate variables, so it's a good idea to put your `fit()` call in the scope as well.\n- Train the model via `fit()` as usual.\n\nImportantly, we recommend that you use `tf.data.Dataset` objects to load data\nin a multi-device or distributed workflow.\n\nSchematically, it looks like this:\n\n```python\n# Create a MirroredStrategy.\nstrategy = tf.distribute.MirroredStrategy()\nprint('Number of devices: {}'.format(strategy.num_replicas_in_sync))\n\n# Open a strategy scope.\nwith strategy.scope():\n    # Everything that creates variables should be under the strategy scope.\n    # In general this is only model construction & `compile()`.\n    model = Model(...)\n    model.compile(...)\n\n    # Train the model on all available devices.\n    model.fit(train_dataset, validation_data=val_dataset, ...)\n\n    # Test the model on all available devices.\n    model.evaluate(test_dataset)\n```\n\nHere's a simple end-to-end runnable example:\n"

def get_compiled_model():
    if False:
        for i in range(10):
            print('nop')
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(256, activation='relu')(inputs)
    x = keras.layers.Dense(256, activation='relu')(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model

def get_dataset():
    if False:
        while True:
            i = 10
    batch_size = 32
    num_val_samples = 10000
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    x_test = x_test.reshape(-1, 784).astype('float32') / 255
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size), tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size), tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size))
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = get_compiled_model()
    (train_dataset, val_dataset, test_dataset) = get_dataset()
    model.fit(train_dataset, epochs=2, validation_data=val_dataset)
    model.evaluate(test_dataset)
"\n## Using callbacks to ensure fault tolerance\n\nWhen using distributed training, you should always make sure you have a strategy to\nrecover from failure (fault tolerance). The simplest way to handle this is to pass\n`ModelCheckpoint` callback to `fit()`, to save your model\nat regular intervals (e.g. every 100 batches or every epoch). You can then restart\ntraining from your saved model.\n\nHere's a simple example:\n"
checkpoint_dir = './ckpt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def make_or_restore_model():
    if False:
        print('Hello World!')
    checkpoints = [checkpoint_dir + '/' + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print('Creating a new model')
    return get_compiled_model()

def run_training(epochs=1):
    if False:
        return 10
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = make_or_restore_model()
        callbacks = [keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + '/ckpt-{epoch}.keras', save_freq='epoch')]
        model.fit(train_dataset, epochs=epochs, callbacks=callbacks, validation_data=val_dataset, verbose=2)
run_training(epochs=1)
run_training(epochs=1)
'\n## `tf.data` performance tips\n\nWhen doing distributed training, the efficiency with which you load data can often become\ncritical. Here are a few tips to make sure your `tf.data` pipelines\nrun as fast as possible.\n\n**Note about dataset batching**\n\nWhen creating your dataset, make sure it is batched with the global batch size.\nFor instance, if each of your 8 GPUs is capable of running a batch of 64 samples, you\ncall use a global batch size of 512.\n\n**Calling `dataset.cache()`**\n\nIf you call `.cache()` on a dataset, its data will be cached after running through the\nfirst iteration over the data. Every subsequent iteration will use the cached data. The\ncache can be in memory (default) or to a local file you specify.\n\nThis can improve performance when:\n\n- Your data is not expected to change from iteration to iteration\n- You are reading data from a remote distributed filesystem\n- You are reading data from local disk, but your data would fit in memory and your\nworkflow is significantly IO-bound (e.g. reading & decoding image files).\n\n**Calling `dataset.prefetch(buffer_size)`**\n\nYou should almost always call `.prefetch(buffer_size)` after creating a dataset. It means\nyour data pipeline will run asynchronously from your model,\nwith new samples being preprocessed and stored in a buffer while the current batch\nsamples are used to train the model. The next batch will be prefetched in GPU memory by\nthe time the current batch is over.\n'
"\nThat's it!\n"