import pytest
from unittest import TestCase
from ... import op_tf2
import numpy as np
import tempfile
import os
from bigdl.chronos.utils import LazyImport
tf = LazyImport('tensorflow')
LSTMModel = LazyImport('bigdl.chronos.model.tf2.VanillaLSTM_keras.LSTMModel')
model_creator = LazyImport('bigdl.chronos.model.tf2.VanillaLSTM_keras.model_creator')

def create_data():
    if False:
        print('Hello World!')
    num_train_samples = 1000
    num_val_samples = 400
    num_test_samples = 400
    input_time_steps = 7
    input_feature_dim = 4
    output_time_steps = 1
    output_feature_dim = np.random.randint(1, 5)

    def get_x_y(num_samples):
        if False:
            print('Hello World!')
        x = np.random.randn(num_samples, input_time_steps, input_feature_dim)
        y = np.random.randn(num_samples, output_time_steps, output_feature_dim)
        return (x, y)
    train_data = get_x_y(num_train_samples)
    val_data = get_x_y(num_val_samples)
    test_data = get_x_y(num_test_samples)
    return (train_data, val_data, test_data)

@op_tf2
class TestVanillaLSTM(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        pass

    def tearDown(self):
        if False:
            return 10
        pass

    def test_lstm_fit_predict_evaluate(self):
        if False:
            return 10
        (train_data, val_data, test_data) = create_data()
        model = model_creator(config={'input_feature_num': 4, 'output_feature_num': test_data[-1].shape[-1]})
        model.fit(train_data[0], train_data[1], epochs=2, validation_data=val_data)
        yhat = model.predict(test_data[0])
        model.evaluate(test_data[0], test_data[1])
        assert yhat.shape == test_data[1].shape

    def test_lstm_save_load(self):
        if False:
            i = 10
            return i + 15
        (train_data, val_data, test_data) = create_data()
        model = model_creator(config={'input_feature_num': 4, 'output_feature_num': test_data[-1].shape[-1]})
        model.fit(train_data[0], train_data[1], epochs=2, validation_data=val_data)
        with tempfile.TemporaryDirectory() as tmp_dir_file:
            model.save(tmp_dir_file)
            import keras
            restore_model = keras.models.load_model(tmp_dir_file, custom_objects={'LSTMModel': LSTMModel})
        model_res = model.evaluate(test_data[0], test_data[1])
        restore_model_res = restore_model.evaluate(test_data[0], test_data[1])
        np.testing.assert_almost_equal(model_res, restore_model_res, decimal=5)
        temp_LSTMModel = LSTMModel(input_dim=4, hidden_dim=[32, 32], layer_num=2, dropout=[0.2, 0.2], output_dim=test_data[-1].shape[-1]).__class__
        assert isinstance(restore_model, temp_LSTMModel)

    def test_lstm_freeze_training(self):
        if False:
            print('Hello World!')
        (train_data, val_data, test_data) = create_data()
        model = model_creator(config={'input_feature_num': 4, 'output_feature_num': test_data[-1].shape[-1]})
        _freeze_yhat = model(test_data[0], training=False)
        freeze_yhat = model(test_data[0], training=False)
        assert np.all(_freeze_yhat == freeze_yhat)
        _unfreeze_yhat = model(test_data[0], training=True)
        unfreeze_yhat = model(test_data[0], training=True)
        assert np.any(_unfreeze_yhat != unfreeze_yhat)
if __name__ == '__main__':
    pytest.main([__file__])