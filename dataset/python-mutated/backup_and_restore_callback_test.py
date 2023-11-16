import numpy as np
import pytest
from keras import callbacks
from keras import layers
from keras import testing
from keras.models import Sequential
from keras.utils import file_utils

class InterruptingCallback(callbacks.Callback):
    """A callback to intentionally interrupt training."""

    def __init__(self, steps_int, epoch_int):
        if False:
            for i in range(10):
                print('nop')
        self.batch_count = 0
        self.epoch_count = 0
        self.steps_int = steps_int
        self.epoch_int = epoch_int

    def on_epoch_end(self, epoch, log=None):
        if False:
            print('Hello World!')
        self.epoch_count += 1
        if self.epoch_int is not None and self.epoch_count == self.epoch_int:
            raise RuntimeError('EpochInterruption')

    def on_batch_end(self, batch, logs=None):
        if False:
            print('Hello World!')
        self.batch_count += 1
        if self.steps_int is not None and self.batch_count == self.steps_int:
            raise RuntimeError('StepsInterruption')

class CanaryLayer(layers.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.counter = self.add_weight(shape=(), initializer='zeros', dtype='float32', trainable=False)

    def call(self, x):
        if False:
            while True:
                i = 10
        self.counter.assign_add(1)
        return x

class BackupAndRestoreCallbackTest(testing.TestCase):

    def make_model(self):
        if False:
            print('Hello World!')
        model = Sequential([CanaryLayer(), layers.Dense(1)])
        model.compile(loss='mse', optimizer='sgd', metrics=['mse'])
        return model

    def test_save_freq_unknown_error(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, expected_regex='Invalid value'):
            callbacks.BackupAndRestore(backup_dir='backup_dir', save_freq='batch')
        with self.assertRaisesRegex(ValueError, expected_regex='Invalid value'):
            callbacks.BackupAndRestore(backup_dir='backup_dir', save_freq=0.15)

    @pytest.mark.requires_trainable_backend
    def test_best_case_step(self):
        if False:
            i = 10
            return i + 15
        temp_dir = self.get_temp_dir()
        backup_dir = file_utils.join(temp_dir, 'subdir')
        self.assertFalse(file_utils.exists(backup_dir))
        model = self.make_model()
        cbk = callbacks.BackupAndRestore(backup_dir, save_freq=1)
        x_train = np.random.random((10, 3))
        y_train = np.random.random((10, 1))
        try:
            model.fit(x_train, y_train, batch_size=4, callbacks=[cbk, InterruptingCallback(steps_int=2, epoch_int=None)], epochs=2, verbose=0)
        except RuntimeError:
            self.assertTrue(file_utils.exists(backup_dir))
            self.assertEqual(cbk._current_epoch, 0)
            self.assertEqual(cbk._last_batch_seen, 1)
            self.assertEqual(int(model.layers[0].counter.value), 2)
            hist = model.fit(x_train, y_train, batch_size=4, callbacks=[cbk], epochs=5)
            self.assertEqual(cbk._current_epoch, 4)
            self.assertEqual(hist.epoch[-1], 4)
            self.assertEqual(int(model.layers[0].counter.value), 17)

    @pytest.mark.requires_trainable_backend
    def test_best_case_epoch(self):
        if False:
            i = 10
            return i + 15
        temp_dir = self.get_temp_dir()
        backup_dir = file_utils.join(temp_dir, 'subdir')
        self.assertFalse(file_utils.exists(backup_dir))
        model = self.make_model()
        self.assertEqual(int(model.layers[0].counter.value), 0)
        cbk = callbacks.BackupAndRestore(backup_dir=backup_dir, save_freq='epoch')
        x_train = np.random.random((10, 3))
        y_train = np.random.random((10, 1))
        try:
            model.fit(x_train, y_train, batch_size=4, callbacks=[cbk, InterruptingCallback(steps_int=None, epoch_int=2)], epochs=6, verbose=0)
        except RuntimeError:
            self.assertEqual(cbk._current_epoch, 1)
            self.assertTrue(file_utils.exists(backup_dir))
            self.assertEqual(int(model.layers[0].counter.value), 6)
            hist = model.fit(x_train, y_train, batch_size=4, callbacks=[cbk], epochs=5)
            self.assertEqual(cbk._current_epoch, 4)
            self.assertEqual(hist.epoch[-1], 4)
            self.assertEqual(int(model.layers[0].counter.value), 21)

    @pytest.mark.requires_trainable_backend
    def test_model_deleted_case_epoch(self):
        if False:
            i = 10
            return i + 15
        temp_dir = self.get_temp_dir()
        backup_dir = file_utils.join(temp_dir, 'subdir')
        self.assertFalse(file_utils.exists(backup_dir))
        model = self.make_model()
        cbk = callbacks.BackupAndRestore(backup_dir, save_freq='epoch')
        x_train = np.random.random((10, 3))
        y_train = np.random.random((10, 1))
        model.fit(x_train, y_train, batch_size=4, callbacks=[cbk], epochs=2, verbose=0)
        self.assertFalse(file_utils.exists(backup_dir))

    def test_backup_dir_empty_error(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, expected_regex='Empty `backup_dir` argument passed'):
            callbacks.BackupAndRestore(backup_dir='', save_freq='epoch')

    def test_backup_dir_none_error(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, expected_regex='Empty `backup_dir` argument passed'):
            callbacks.BackupAndRestore(backup_dir=None, save_freq='epoch')