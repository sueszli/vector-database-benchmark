import tempfile
import numpy as np
import pytest
from ... import op_tf2
from unittest import TestCase
from bigdl.chronos.utils import LazyImport
tf = LazyImport('tensorflow')
model_creator = LazyImport('bigdl.chronos.model.tf2.TCN_keras.model_creator')
TemporalConvNet = LazyImport('bigdl.chronos.model.tf2.TCN_keras.TemporalConvNet')
TemporalBlock = LazyImport('bigdl.chronos.model.tf2.TCN_keras.TemporalBlock')

def create_data():
    if False:
        print('Hello World!')
    train_num_samples = 1000
    test_num_samples = 400
    past_seq_len = 20
    input_feature_num = 10
    output_feature_num = 2
    future_seq_len = 10

    def get_x_y(num_samples):
        if False:
            while True:
                i = 10
        x = np.random.randn(num_samples, past_seq_len, input_feature_num)
        y = np.random.randn(num_samples, future_seq_len, output_feature_num)
        return (x, y)
    train_data = get_x_y(train_num_samples)
    test_data = get_x_y(test_num_samples)
    return (train_data, test_data)

@op_tf2
class TestTcnKeras(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def tearDown(self):
        if False:
            return 10
        pass

    def test_tcn_fit_predict_evaluate(self):
        if False:
            for i in range(10):
                print('nop')
        (train_data, test_data) = create_data()
        model = model_creator(config={'past_seq_len': 20, 'future_seq_len': 10, 'input_feature_num': 10, 'output_feature_num': 2})
        model.fit(train_data[0], train_data[1], epochs=2, validation_data=test_data)
        yhat = model.predict(test_data[0])
        model.evaluate(test_data[0], test_data[1])
        assert yhat.shape == test_data[1].shape

    def test_tcn_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        (train_data, test_data) = create_data()
        model = model_creator(config={'past_seq_len': 20, 'future_seq_len': 10, 'input_feature_num': 10, 'output_feature_num': 2})
        model.fit(train_data[0], train_data[1], epochs=2, validation_data=test_data)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            model.save(tmp_dir_name)
            restore_model = tf.keras.models.load_model(tmp_dir_name, custom_objects={'TemporalConvNet': TemporalConvNet, 'TemporalBlock': TemporalBlock})
        model_res = model.evaluate(test_data[0], test_data[1])
        restore_model_res = restore_model.evaluate(test_data[0], test_data[1])
        np.testing.assert_almost_equal(model_res, restore_model_res, decimal=5)
        temp_TemporalConvNet = TemporalConvNet(past_seq_len=20, future_seq_len=10, input_feature_num=10, output_feature_num=2).__class__
        assert isinstance(restore_model, temp_TemporalConvNet)
if __name__ == '__main__':
    pytest.main([__file__])