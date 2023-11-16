import os.path
import tempfile
import unittest
from typing import List
import pytest
import tensorflow as tf
from numpy import ndarray
from ray import train
from ray.data import Preprocessor
from ray.train import ScalingConfig
from ray.train.tensorflow import TensorflowCheckpoint, TensorflowTrainer

class DummyPreprocessor(Preprocessor):

    def __init__(self, multiplier):
        if False:
            while True:
                i = 10
        self.multiplier = multiplier

    def transform_batch(self, df):
        if False:
            while True:
                i = 10
        return df * self.multiplier

def get_model():
    if False:
        while True:
            i = 10
    return tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=()), tf.keras.layers.Flatten(), tf.keras.layers.Dense(10), tf.keras.layers.Dense(1)])

def compare_weights(w1: List[ndarray], w2: List[ndarray]) -> bool:
    if False:
        return 10
    if not len(w1) == len(w2):
        return False
    size = len(w1)
    for i in range(size):
        comparison = w1[i] == w2[i]
        if not comparison.all():
            return False
    return True

class TestFromModel(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model = get_model()
        self.preprocessor = DummyPreprocessor(1)

    def test_from_model(self):
        if False:
            return 10
        checkpoint = TensorflowCheckpoint.from_model(self.model, preprocessor=DummyPreprocessor(1))
        loaded_model = checkpoint.get_model()
        preprocessor = checkpoint.get_preprocessor()
        assert compare_weights(loaded_model.get_weights(), self.model.get_weights())
        assert preprocessor.multiplier == 1

    def test_from_saved_model(self):
        if False:
            return 10
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir_path = os.path.join(tmp_dir, 'my_model')
            self.model.save(model_dir_path, save_format='tf')
            checkpoint = TensorflowCheckpoint.from_saved_model(model_dir_path, preprocessor=DummyPreprocessor(1))
            loaded_model = checkpoint.get_model()
            preprocessor = checkpoint.get_preprocessor()
            assert compare_weights(self.model.get_weights(), loaded_model.get_weights())
            assert preprocessor.multiplier == 1

    def test_from_h5_model(self):
        if False:
            return 10
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file_path = os.path.join(tmp_dir, 'my_model.h5')
            self.model.save(model_file_path)
            checkpoint = TensorflowCheckpoint.from_h5(model_file_path, preprocessor=DummyPreprocessor(1))
            loaded_model = checkpoint.get_model()
            preprocessor = checkpoint.get_preprocessor()
            assert compare_weights(self.model.get_weights(), loaded_model.get_weights())
            assert preprocessor.multiplier == 1

def test_tensorflow_checkpoint_saved_model():
    if False:
        i = 10
        return i + 15

    def train_fn():
        if False:
            for i in range(10):
                print('nop')
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=()), tf.keras.layers.Flatten(), tf.keras.layers.Dense(10), tf.keras.layers.Dense(1)])
        with tempfile.TemporaryDirectory() as tempdir:
            model.save(tempdir)
            checkpoint = TensorflowCheckpoint.from_saved_model(tempdir)
            train.report({'my_metric': 1}, checkpoint=checkpoint)
    trainer = TensorflowTrainer(train_loop_per_worker=train_fn, scaling_config=ScalingConfig(num_workers=2))
    assert trainer.fit().checkpoint

def test_tensorflow_checkpoint_h5():
    if False:
        print('Hello World!')

    def train_func():
        if False:
            while True:
                i = 10
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=()), tf.keras.layers.Flatten(), tf.keras.layers.Dense(10), tf.keras.layers.Dense(1)])
        with tempfile.TemporaryDirectory() as tempdir:
            model.save(os.path.join(tempdir, 'my_model.h5'))
            checkpoint = TensorflowCheckpoint.from_h5(os.path.join(tempdir, 'my_model.h5'))
            train.report({'my_metric': 1}, checkpoint=checkpoint)
    trainer = TensorflowTrainer(train_loop_per_worker=train_func, scaling_config=ScalingConfig(num_workers=2))
    assert trainer.fit().checkpoint
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', '-x', __file__]))