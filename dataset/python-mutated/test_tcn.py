from unittest import TestCase
from bigdl.chronos.utils import LazyImport
TCNPytorch = LazyImport('bigdl.chronos.model.tcn.TCNPytorch')
import numpy as np
import tempfile
import os
from .. import op_torch, op_distributed

def create_data():
    if False:
        for i in range(10):
            print('nop')
    num_train_samples = 1000
    num_val_samples = 400
    num_test_samples = 400
    input_time_steps = 24
    input_feature_dim = 4
    output_time_steps = np.random.randint(1, 5)
    output_feature_dim = np.random.randint(1, 4)

    def get_x_y(num_samples):
        if False:
            i = 10
            return i + 15
        x = np.random.rand(num_samples, input_time_steps, input_feature_dim)
        y = np.random.randn(num_samples, output_time_steps, output_feature_dim)
        return (x, y)
    train_data = get_x_y(num_train_samples)
    val_data = get_x_y(num_val_samples)
    test_data = get_x_y(num_test_samples)
    return (train_data, val_data, test_data)

@op_torch
@op_distributed
class TestTcn(TestCase):
    (train_data, val_data, test_data) = create_data()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.model = TCNPytorch()

    def tearDown(self):
        if False:
            print('Hello World!')
        del self.model

    def test_fit_evaluate(self):
        if False:
            print('Hello World!')
        config = {'batch_size': 128}
        self.model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data, metric='mse', **config)
        (mse, smape) = self.model.evaluate(self.val_data[0], self.val_data[1], metrics=['mse', 'smape'])
        assert len(mse) == self.val_data[1].shape[-2]
        assert len(mse[0]) == self.val_data[1].shape[-1]
        assert len(smape) == self.val_data[1].shape[-2]
        assert len(smape[0]) == self.val_data[1].shape[-1]

    def test_predict_save_restore(self):
        if False:
            while True:
                i = 10
        config = {'batch_size': 128}
        self.model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data, metric='mse', **config)
        pred = self.model.predict(self.test_data[0])
        assert pred.shape == self.test_data[1].shape
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, 'ckpt')
            self.model.save(ckpt_name)
            model_1 = TCNPytorch()
            model_1.restore(ckpt_name)
            pred_1 = model_1.predict(self.test_data[0])
            assert np.allclose(pred, pred_1)