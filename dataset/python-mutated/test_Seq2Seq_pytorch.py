from unittest import TestCase
from .. import op_torch, op_distributed
import numpy as np
import tempfile
import os
import random
from bigdl.chronos.utils import LazyImport
Seq2SeqPytorch = LazyImport('bigdl.chronos.model.Seq2Seq_pytorch.Seq2SeqPytorch')

def create_data():
    if False:
        i = 10
        return i + 15
    num_train_samples = 1000
    num_val_samples = 400
    num_test_samples = 400
    input_time_steps = random.randint(20, 30)
    input_feature_dim = random.randint(4, 5)
    output_time_steps = random.randint(10, 30)
    output_feature_dim = random.randint(1, 3)

    def get_x_y(num_samples):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.rand(num_samples, input_time_steps, input_feature_dim)
        y = np.random.rand(num_samples, output_time_steps, output_feature_dim)
        return (x, y)
    train_data = get_x_y(num_train_samples)
    val_data = get_x_y(num_val_samples)
    test_data = get_x_y(num_test_samples)
    return (train_data, val_data, test_data)

@op_torch
@op_distributed
class TestSeq2SeqPytorch(TestCase):
    (train_data, val_data, test_data) = create_data()

    def test_s2s_fit_evaluate(self):
        if False:
            return 10
        model = Seq2SeqPytorch()
        config = {'batch_size': 128, 'teacher_forcing': False}
        model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data, metric='mse', **config)
        (mse, smape) = model.evaluate(self.val_data[0], self.val_data[1], metrics=['mse', 'smape'])
        assert len(mse) == self.val_data[1].shape[-2]
        assert len(mse[0]) == self.val_data[1].shape[-1]
        assert len(smape) == self.val_data[1].shape[-2]
        assert len(smape[0]) == self.val_data[1].shape[-1]

    def test_s2s_teacher_forcing_fit_evaluate(self):
        if False:
            while True:
                i = 10
        model = Seq2SeqPytorch()
        config = {'batch_size': 128, 'teacher_forcing': True}
        model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data, metric='mse', **config)
        (mse, smape) = model.evaluate(self.val_data[0], self.val_data[1], metrics=['mse', 'smape'])
        assert len(mse) == self.val_data[1].shape[-2]
        assert len(mse[0]) == self.val_data[1].shape[-1]
        assert len(smape) == self.val_data[1].shape[-2]
        assert len(smape[0]) == self.val_data[1].shape[-1]

    def test_s2s_predict_save_restore(self):
        if False:
            while True:
                i = 10
        model = Seq2SeqPytorch()
        config = {'batch_size': 128}
        model.fit_eval((self.train_data[0], self.train_data[1]), self.val_data, metric='mse', **config)
        pred = model.predict(self.test_data[0])
        assert pred.shape == self.test_data[1].shape
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, 'ckpt')
            model.save(ckpt_name)
            model_1 = Seq2SeqPytorch()
            model_1.restore(ckpt_name)
            pred_1 = model_1.predict(self.test_data[0])
            assert np.allclose(pred, pred_1)