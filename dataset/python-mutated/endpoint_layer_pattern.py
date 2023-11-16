"""
Title: Endpoint layer pattern
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2019/05/10
Last modified: 2019/05/10
Description: Demonstration of the "endpoint layer" pattern (layer that handles loss management).
Accelerator: GPU
"""
'\n## Setup\n'
import tensorflow as tf
import keras
import numpy as np
'\n## Usage of endpoint layers in the Functional API\n\nAn "endpoint layer" has access to the model\'s targets, and creates arbitrary losses\nin `call()` using `self.add_loss()` and `Metric.update_state()`.\nThis enables you to define losses and\nmetrics that don\'t match the usual signature `fn(y_true, y_pred, sample_weight=None)`.\n\nNote that you could have separate metrics for training and eval with this pattern.\n'

class LogisticEndpoint(keras.layers.Layer):

    def __init__(self, name=None):
        if False:
            print('Hello World!')
        super().__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_metric = keras.metrics.BinaryAccuracy(name='accuracy')

    def call(self, logits, targets=None, sample_weight=None):
        if False:
            print('Hello World!')
        if targets is not None:
            loss = self.loss_fn(targets, logits, sample_weight)
            self.add_loss(loss)
            self.accuracy_metric.update_state(targets, logits, sample_weight)
        return tf.nn.softmax(logits)
inputs = keras.Input((764,), name='inputs')
logits = keras.layers.Dense(1)(inputs)
targets = keras.Input((1,), name='targets')
sample_weight = keras.Input((1,), name='sample_weight')
preds = LogisticEndpoint()(logits, targets, sample_weight)
model = keras.Model([inputs, targets, sample_weight], preds)
data = {'inputs': np.random.random((1000, 764)), 'targets': np.random.random((1000, 1)), 'sample_weight': np.random.random((1000, 1))}
model.compile(keras.optimizers.Adam(0.001))
model.fit(data, epochs=2)
"\n## Exporting an inference-only model\n\nSimply don't include `targets` in the model. The weights stay the same.\n"
inputs = keras.Input((764,), name='inputs')
logits = keras.layers.Dense(1)(inputs)
preds = LogisticEndpoint()(logits, targets=None, sample_weight=None)
inference_model = keras.Model(inputs, preds)
inference_model.set_weights(model.get_weights())
preds = inference_model.predict(np.random.random((1000, 764)))
'\n## Usage of loss endpoint layers in subclassed models\n'

class LogReg(keras.Model):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.dense = keras.layers.Dense(1)
        self.logistic_endpoint = LogisticEndpoint()

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        logits = self.dense(inputs['inputs'])
        preds = self.logistic_endpoint(logits=logits, targets=inputs['targets'], sample_weight=inputs['sample_weight'])
        return preds
model = LogReg()
data = {'inputs': np.random.random((1000, 764)), 'targets': np.random.random((1000, 1)), 'sample_weight': np.random.random((1000, 1))}
model.compile(keras.optimizers.Adam(0.001))
model.fit(data, epochs=2)