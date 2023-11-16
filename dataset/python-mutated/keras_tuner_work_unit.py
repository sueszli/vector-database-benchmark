"""A work unit for training, evaluating, and saving a Keras model."""
import os
import time
from adanet.experimental.work_units import work_unit
from kerastuner.engine.tuner import Tuner
import tensorflow.compat.v2 as tf

class KerasTunerWorkUnit(work_unit.WorkUnit):
    """Trains, evaluates and saves a tuned Keras model."""

    def __init__(self, tuner: Tuner, *search_args, **search_kwargs):
        if False:
            print('Hello World!')
        self._tuner = tuner
        self._search_args = search_args
        self._search_kwargs = search_kwargs

    def execute(self):
        if False:
            while True:
                i = 10
        log_dir = os.path.join('/tmp', str(int(time.time())))
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
        self._tuner.search(*self._search_args, callbacks=[tensorboard], **self._search_kwargs)