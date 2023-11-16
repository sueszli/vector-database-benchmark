"""A work unit for training, evaluating, and saving a Keras model."""
import os
import time
from adanet.experimental.storages.storage import ModelContainer
from adanet.experimental.storages.storage import Storage
from adanet.experimental.work_units import work_unit
import tensorflow.compat.v2 as tf

class KerasTrainerWorkUnit(work_unit.WorkUnit):
    """Trains, evaluates, and saves a Keras model."""

    def __init__(self, model: tf.keras.Model, train_dataset: tf.data.Dataset, eval_dataset: tf.data.Dataset, storage: Storage, tensorboard_base_dir: str='/tmp'):
        if False:
            for i in range(10):
                print('nop')
        self._model = model
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._storage = storage
        self._tensorboard_base_dir = tensorboard_base_dir

    def execute(self):
        if False:
            for i in range(10):
                print('nop')
        log_dir = os.path.join(self._tensorboard_base_dir, str(int(time.time())))
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
        if self._model.trainable:
            self._model.fit(self._train_dataset, callbacks=[tensorboard])
        else:
            print('Skipping training since model.trainable set to false.')
        results = self._model.evaluate(self._eval_dataset, callbacks=[tensorboard])
        if not isinstance(results, list):
            results = [results]
        self._storage.save_model(ModelContainer(results[0], self._model, results))