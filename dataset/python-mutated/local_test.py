import os
import subprocess
import tempfile
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import data_utils
import predict
import trainer

def test_validated_missing_field() -> None:
    if False:
        while True:
            i = 10
    tensor_dict = {}
    values_spec = {'x': tf.TensorSpec(shape=(3,), dtype=tf.float32)}
    with pytest.raises(KeyError):
        trainer.validated(tensor_dict, values_spec)

def test_validated_incompatible_type() -> None:
    if False:
        while True:
            i = 10
    tensor_dict = {'x': tf.constant(['a', 'b', 'c'])}
    values_spec = {'x': tf.TensorSpec(shape=(3,), dtype=tf.float32)}
    with pytest.raises(TypeError):
        trainer.validated(tensor_dict, values_spec)

def test_validated_incompatible_shape() -> None:
    if False:
        i = 10
        return i + 15
    tensor_dict = {'x': tf.constant([1.0])}
    values_spec = {'x': tf.TensorSpec(shape=(3,), dtype=tf.float32)}
    with pytest.raises(ValueError):
        trainer.validated(tensor_dict, values_spec)

def test_validated_ok() -> None:
    if False:
        return 10
    tensor_dict = {'x': tf.constant([1.0, 2.0, 3.0])}
    values_spec = {'x': tf.TensorSpec(shape=(3,), dtype=tf.float32)}
    trainer.validated(tensor_dict, values_spec)
    tensor_dict = {'x': tf.constant([[1.0], [2.0], [3.0]])}
    values_spec = {'x': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)}
    trainer.validated(tensor_dict, values_spec)

def test_serialize_deserialize() -> None:
    if False:
        print('Hello World!')
    unlabeled_data = data_utils.read_data('test_data/56980685061237.npz')
    labels = data_utils.read_labels('test_data/labels.csv')
    data = data_utils.label_data(unlabeled_data, labels)
    for training_point in data_utils.generate_training_points(data):
        serialized = trainer.serialize(training_point)
        (inputs, outputs) = trainer.deserialize(serialized)
        assert set(inputs.keys()) == set(trainer.INPUTS_SPEC.keys())
        assert set(outputs.keys()) == set(trainer.OUTPUTS_SPEC.keys())

@mock.patch.object(trainer, 'PADDING', 2)
def test_e2e_local() -> None:
    if False:
        while True:
            i = 10
    with tempfile.TemporaryDirectory() as temp_dir:
        train_data_dir = os.path.join(temp_dir, 'datasets', 'train')
        eval_data_dir = os.path.join(temp_dir, 'datasets', 'eval')
        model_dir = os.path.join(temp_dir, 'model')
        tensorboard_dir = os.path.join(temp_dir, 'tensorboard')
        checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        cmd = ['python', 'create_datasets.py', '--raw-data-dir=test_data', '--raw-labels-dir=test_data', f'--train-data-dir={train_data_dir}', f'--eval-data-dir={eval_data_dir}']
        subprocess.run(cmd, check=True)
        assert os.listdir(train_data_dir), 'no training files found'
        assert os.listdir(eval_data_dir), 'no evaluation files found'
        trainer.run(train_data_dir=train_data_dir, eval_data_dir=eval_data_dir, model_dir=model_dir, tensorboard_dir=tensorboard_dir, checkpoint_dir=checkpoint_dir, train_epochs=2, batch_size=8)
        assert os.listdir(model_dir), 'no model files found'
        assert os.listdir(tensorboard_dir), 'no tensorboard files found'
        assert os.listdir(checkpoint_dir), 'no checkpoint files found'
        with open('test_data/56980685061237.npz', 'rb') as f:
            input_data = pd.DataFrame(np.load(f)['x'])
        predictions = predict.run(model_dir, input_data.to_dict('list'))
        assert 'is_fishing' in predictions
        assert len(predictions['is_fishing']) > 0