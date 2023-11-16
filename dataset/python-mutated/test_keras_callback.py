from typing import Dict, Tuple
from unittest.mock import patch
import os
import pytest
import numpy as np
import tensorflow as tf
import ray
from ray import train
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.train.constants import TRAIN_DATASET_KEY
from ray.train import ScalingConfig
from ray.train.tensorflow import TensorflowTrainer, TensorflowPredictor, TensorflowCheckpoint

class TestReportCheckpointCallback:

    @pytest.fixture(name='model')
    def model_fixture(self):
        if False:
            for i in range(10):
                print('nop')
        model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(1,)), tf.keras.layers.Dense(1)])
        model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
        return model

    @patch('ray.train.report')
    @pytest.mark.parametrize('metrics, expected_metrics_keys', [(None, {'loss', 'accuracy', 'val_loss', 'val_accuracy'}), ('loss', {'loss'}), (['loss', 'accuracy'], {'loss', 'accuracy'}), ({'spam': 'loss'}, {'spam'})])
    def test_reported_metrics_contain_expected_keys(self, mock_report, metrics, expected_metrics_keys, model):
        if False:
            print('Hello World!')
        model.fit(x=np.zeros((1, 1)), y=np.zeros((1, 1)), validation_data=(np.zeros((1, 1)), np.zeros((1, 1))), callbacks=[ReportCheckpointCallback(metrics=metrics)])
        for ((metrics,), _) in ray.train.report.call_args_list:
            assert metrics.keys() == expected_metrics_keys

    @patch('ray.train.report')
    def test_report_with_default_arguments(self, mock_report, model):
        if False:
            for i in range(10):
                print('nop')
        callback = ReportCheckpointCallback()
        callback.model = model
        callback.on_epoch_end(0, {'loss': 0})
        assert len(ray.train.report.call_args_list) == 1
        (metrics, checkpoint) = self.parse_call(ray.train.report.call_args_list[0])
        assert metrics == {'loss': 0}
        assert checkpoint is not None

    @patch('ray.train.report')
    def test_checkpoint_on_list(self, mock_report, model):
        if False:
            i = 10
            return i + 15
        callback = ReportCheckpointCallback(checkpoint_on=['epoch_end', 'train_batch_end'])
        callback.model = model
        callback.on_train_batch_end(0, {'loss': 0})
        callback.on_epoch_end(0, {'loss': 0})
        assert len(ray.train.report.call_args_list) == 2
        (_, first_checkpoint) = self.parse_call(ray.train.report.call_args_list[0])
        assert first_checkpoint is not None
        (_, second_checkpoint) = self.parse_call(ray.train.report.call_args_list[0])
        assert second_checkpoint is not None

    @patch('ray.train.report')
    def test_report_metrics_on_list(self, mock_report, model):
        if False:
            print('Hello World!')
        callback = ReportCheckpointCallback(report_metrics_on=['epoch_end', 'train_batch_end'])
        callback.model = model
        callback.on_train_batch_end(0, {'loss': 0})
        callback.on_epoch_end(0, {'loss': 1})
        assert len(ray.train.report.call_args_list) == 2
        (first_metric, _) = self.parse_call(ray.train.report.call_args_list[0])
        assert first_metric == {'loss': 0}
        (second_metric, _) = self.parse_call(ray.train.report.call_args_list[1])
        assert second_metric == {'loss': 1}

    @patch('ray.train.report')
    def test_report_and_checkpoint_on_different_events(self, mock_report, model):
        if False:
            print('Hello World!')
        callback = ReportCheckpointCallback(report_metrics_on='train_batch_end', checkpoint_on='epoch_end')
        callback.model = model
        callback.on_train_batch_end(0, {'loss': 0})
        callback.on_epoch_end(0, {'loss': 1})
        assert len(ray.train.report.call_args_list) == 2
        (first_metric, first_checkpoint) = self.parse_call(ray.train.report.call_args_list[0])
        assert first_metric == {'loss': 0}
        assert first_checkpoint is None
        (second_metric, second_checkpoint) = self.parse_call(ray.train.report.call_args_list[1])
        assert second_metric == {'loss': 1}
        assert second_checkpoint is not None

    @patch('ray.train.report')
    def test_report_delete_tempdir(self, mock_report, model):
        if False:
            return 10
        callback = ReportCheckpointCallback()
        callback.model = model
        callback.on_epoch_end(0, {'loss': 0})
        assert len(ray.train.report.call_args_list) == 1
        (metrics, checkpoint) = self.parse_call(ray.train.report.call_args_list[0])
        assert metrics == {'loss': 0}
        assert checkpoint is not None
        assert checkpoint.path is not None
        assert not os.path.exists(checkpoint.path)

    def parse_call(self, call) -> Tuple[Dict, train.Checkpoint]:
        if False:
            i = 10
            return i + 15
        ((metrics,), kwargs) = call
        checkpoint = kwargs['checkpoint']
        return (metrics, checkpoint)

def get_dataset(a=5, b=10, size=1000):
    if False:
        while True:
            i = 10
    items = [i / size for i in range(size)]
    dataset = ray.data.from_items([{'x': x, 'y': a * x + b} for x in items])
    return dataset

def build_model() -> tf.keras.Model:
    if False:
        for i in range(10):
            print('nop')
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=()), tf.keras.layers.Flatten(), tf.keras.layers.Dense(10), tf.keras.layers.Dense(1)])
    return model

def train_func(config: dict):
    if False:
        i = 10
        return i + 15
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        multi_worker_model = build_model()
        multi_worker_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=config.get('lr', 0.001)), loss=tf.keras.losses.mean_squared_error, metrics=[tf.keras.metrics.mean_squared_error])
    dataset = train.get_dataset_shard('train')
    for _ in range(config.get('epoch', 3)):
        tf_dataset = dataset.to_tf('x', 'y', batch_size=32)
        multi_worker_model.fit(tf_dataset, callbacks=[ReportCheckpointCallback()])

def test_keras_callback_e2e():
    if False:
        while True:
            i = 10
    epochs = 3
    config = {'epochs': epochs}
    trainer = TensorflowTrainer(train_loop_per_worker=train_func, train_loop_config=config, scaling_config=ScalingConfig(num_workers=2), datasets={TRAIN_DATASET_KEY: get_dataset()})
    checkpoint = trainer.fit().checkpoint
    tf_checkpoint = TensorflowCheckpoint(path=checkpoint.path, filesystem=checkpoint.filesystem)
    predictor = TensorflowPredictor.from_checkpoint(tf_checkpoint)
    items = np.random.uniform(0, 1, size=(10, 1))
    predictor.predict(data=items)
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(['-v', '-x', __file__]))