from bigdl.chronos.utils import LazyImport
torch = LazyImport('torch')
tf = LazyImport('tensorflow')
import numpy as np
from unittest import TestCase
import pytest
import tempfile
from ... import op_torch, op_tf2, op_distributed, op_inference
AutoTCN = LazyImport('bigdl.chronos.autots.model.auto_tcn.AutoTCN')
hp = LazyImport('bigdl.orca.automl.hp')
input_feature_dim = 10
output_feature_dim = 2
past_seq_len = 5
future_seq_len = 1

def get_x_y(size):
    if False:
        return 10
    x = np.random.randn(size, past_seq_len, input_feature_dim)
    y = np.random.randn(size, future_seq_len, output_feature_dim)
    return (x.astype(np.float32), y.astype(np.float32))

def gen_RandomDataset():
    if False:
        i = 10
        return i + 15
    import torch
    from torch.utils.data import Dataset

    class RandomDataset(Dataset):

        def __init__(self, size=1000):
            if False:
                while True:
                    i = 10
            (x, y) = get_x_y(size)
            self.x = torch.from_numpy(x).float()
            self.y = torch.from_numpy(y).float()

        def __len__(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.x.shape[0]

        def __getitem__(self, idx):
            if False:
                while True:
                    i = 10
            return (self.x[idx], self.y[idx])
    return RandomDataset

def train_dataloader_creator(config):
    if False:
        for i in range(10):
            print('nop')
    import torch
    from torch.utils.data import DataLoader
    RandomDataset = gen_RandomDataset()
    return DataLoader(RandomDataset(size=1000), batch_size=config['batch_size'], shuffle=True)

def valid_dataloader_creator(config):
    if False:
        return 10
    import torch
    from torch.utils.data import DataLoader
    RandomDataset = gen_RandomDataset()
    return DataLoader(RandomDataset(size=400), batch_size=config['batch_size'], shuffle=True)

def get_auto_estimator(backend='torch'):
    if False:
        while True:
            i = 10
    loss = 'mse' if backend.startswith('keras') else torch.nn.MSELoss()
    auto_tcn = AutoTCN(input_feature_num=input_feature_dim, output_target_num=output_feature_dim, past_seq_len=past_seq_len, future_seq_len=future_seq_len, optimizer='Adam', loss=loss, metric='mse', backend=backend, hidden_units=8, num_channels=[16] * 2, levels=hp.randint(1, 3), kernel_size=hp.choice([2, 3]), lr=hp.choice([0.001, 0.003, 0.01]), dropout=hp.uniform(0.1, 0.2), logs_dir='/tmp/auto_tcn', cpus_per_trial=2, name='auto_tcn')
    return auto_tcn

@op_distributed
class TestAutoTCN(TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        from bigdl.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        from bigdl.orca import stop_orca_context
        stop_orca_context()

    @op_torch
    def test_fit_np(self):
        if False:
            for i in range(10):
                print('nop')
        auto_tcn = get_auto_estimator()
        auto_tcn.fit(data=get_x_y(size=1000), epochs=1, batch_size=hp.choice([32, 64]), validation_data=get_x_y(size=400), n_sampling=1)
        assert auto_tcn.get_best_model()
        best_config = auto_tcn.get_best_config()
        assert 0.1 <= best_config['dropout'] <= 0.2
        assert best_config['batch_size'] in (32, 64)
        assert 1 <= best_config['levels'] < 3

    @op_tf2
    def test_fit_np_keras(self):
        if False:
            for i in range(10):
                print('nop')
        keras_auto_tcn = get_auto_estimator('keras')
        keras_auto_tcn.fit(data=get_x_y(size=1000), epochs=2, batch_size=hp.choice([32, 64]), validation_data=get_x_y(size=400), n_sampling=1)
        assert keras_auto_tcn.get_best_model()
        best_config = keras_auto_tcn.get_best_config()
        assert 0.1 <= best_config['dropout'] <= 0.2
        assert best_config['batch_size'] in (32, 64)
        assert 1 <= best_config['levels'] < 3

    @op_torch
    def test_fit_loader(self):
        if False:
            print('Hello World!')
        auto_tcn = get_auto_estimator()
        auto_tcn.fit(data=train_dataloader_creator(config={'batch_size': 64}), epochs=1, validation_data=valid_dataloader_creator(config={'batch_size': 64}), n_sampling=1)
        assert auto_tcn.get_best_model()
        best_config = auto_tcn.get_best_config()
        assert 0.1 <= best_config['dropout'] <= 0.2
        assert 1 <= best_config['levels'] < 3

    @op_torch
    def test_fit_data_creator(self):
        if False:
            return 10
        auto_tcn = get_auto_estimator()
        auto_tcn.fit(data=train_dataloader_creator, epochs=1, batch_size=hp.choice([32, 64]), validation_data=valid_dataloader_creator, n_sampling=1)
        assert auto_tcn.get_best_model()
        best_config = auto_tcn.get_best_config()
        assert 0.1 <= best_config['dropout'] <= 0.2
        assert best_config['batch_size'] in (32, 64)
        assert 1 <= best_config['levels'] < 3

    @op_torch
    def test_num_channels(self):
        if False:
            i = 10
            return i + 15
        auto_tcn = AutoTCN(input_feature_num=input_feature_dim, output_target_num=output_feature_dim, past_seq_len=past_seq_len, future_seq_len=future_seq_len, optimizer='Adam', loss=torch.nn.MSELoss(), metric='mse', hidden_units=4, levels=hp.randint(1, 3), num_channels=[8] * 2, kernel_size=hp.choice([2, 3]), lr=hp.choice([0.001, 0.003, 0.01]), dropout=hp.uniform(0.1, 0.2), logs_dir='/tmp/auto_tcn', cpus_per_trial=2, name='auto_tcn')
        auto_tcn.fit(data=train_dataloader_creator, epochs=1, batch_size=hp.choice([32, 64]), validation_data=valid_dataloader_creator, n_sampling=1)
        assert auto_tcn.get_best_model()
        best_config = auto_tcn.get_best_config()
        assert best_config['num_channels'] == [8] * 2

    @op_torch
    def test_predict_evaluation(self):
        if False:
            while True:
                i = 10
        auto_tcn = get_auto_estimator()
        auto_tcn.fit(data=train_dataloader_creator(config={'batch_size': 64}), epochs=1, validation_data=valid_dataloader_creator(config={'batch_size': 64}), n_sampling=1)
        (test_data_x, test_data_y) = get_x_y(size=100)
        auto_tcn.predict(test_data_x)
        auto_tcn.evaluate((test_data_x, test_data_y))

    @op_torch
    @op_inference
    def test_onnx_methods(self):
        if False:
            return 10
        auto_tcn = get_auto_estimator()
        auto_tcn.fit(data=train_dataloader_creator(config={'batch_size': 64}), epochs=1, validation_data=valid_dataloader_creator(config={'batch_size': 64}), n_sampling=1)
        (test_data_x, test_data_y) = get_x_y(size=100)
        pred = auto_tcn.predict(test_data_x)
        eval_res = auto_tcn.evaluate((test_data_x, test_data_y))
        try:
            import onnx
            import onnxruntime
            pred_onnx = auto_tcn.predict_with_onnx(test_data_x)
            eval_res_onnx = auto_tcn.evaluate_with_onnx((test_data_x, test_data_y))
            np.testing.assert_almost_equal(pred, pred_onnx, decimal=5)
            np.testing.assert_almost_equal(eval_res, eval_res_onnx, decimal=5)
        except ImportError:
            pass

    @op_torch
    @op_inference
    def test_save_load(self):
        if False:
            i = 10
            return i + 15
        auto_tcn = get_auto_estimator()
        auto_tcn.fit(data=train_dataloader_creator(config={'batch_size': 64}), epochs=1, validation_data=valid_dataloader_creator(config={'batch_size': 64}), n_sampling=1)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            auto_tcn.save(tmp_dir_name)
            auto_tcn.load(tmp_dir_name)
        (test_data_x, test_data_y) = get_x_y(size=100)
        pred = auto_tcn.predict(test_data_x)
        eval_res = auto_tcn.evaluate((test_data_x, test_data_y))
        try:
            import onnx
            import onnxruntime
            pred_onnx = auto_tcn.predict_with_onnx(test_data_x)
            eval_res_onnx = auto_tcn.evaluate_with_onnx((test_data_x, test_data_y))
            np.testing.assert_almost_equal(pred, pred_onnx, decimal=5)
            np.testing.assert_almost_equal(eval_res, eval_res_onnx, decimal=5)
        except ImportError:
            pass

    @op_tf2
    def test_save_load_keras(self):
        if False:
            print('Hello World!')
        auto_keras_tcn = get_auto_estimator(backend='keras')
        auto_keras_tcn.fit(data=get_x_y(size=1000), epochs=2, batch_size=hp.choice([32, 64]), validation_data=get_x_y(size=400), n_sampling=1)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            auto_keras_tcn.save(tmp_dir_name)
            auto_keras_tcn.load(tmp_dir_name)
        (test_data_x, test_data_y) = get_x_y(size=100)
        pred = auto_keras_tcn.predict(test_data_x)
        eval_res = auto_keras_tcn.evaluate((test_data_x, test_data_y))
if __name__ == '__main__':
    pytest.main([__file__])