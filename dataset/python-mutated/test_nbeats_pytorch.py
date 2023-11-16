from unittest import TestCase
from bigdl.chronos.utils import LazyImport
model_creator = LazyImport('bigdl.chronos.model.nbeats_pytorch.model_creator')
Trainer = LazyImport('bigdl.nano.pytorch.trainer.Trainer')
torch = LazyImport('torch')
import numpy as np
import tempfile
import os
from .. import op_torch

def create_data(loader=False):
    if False:
        i = 10
        return i + 15
    num_train_samples = 1000
    num_val_samples = 400
    num_test_samples = 400
    input_time_steps = 24
    input_feature_dim = 1
    output_time_steps = 5
    output_feature_dim = 1

    def get_x_y(num_samples):
        if False:
            i = 10
            return i + 15
        x = np.random.rand(num_samples, input_time_steps, input_feature_dim).astype(np.float32)
        y = x[:, -output_time_steps:, :] * 2 + np.random.rand(num_samples, output_time_steps, output_feature_dim).astype(np.float32)
        return (x, y)
    train_data = get_x_y(num_train_samples)
    val_data = get_x_y(num_val_samples)
    test_data = get_x_y(num_test_samples)
    if loader:
        from torch.utils.data import DataLoader, TensorDataset
        train_loader = DataLoader(TensorDataset(torch.from_numpy(train_data[0]), torch.from_numpy(train_data[1])), batch_size=32)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(val_data[0]), torch.from_numpy(val_data[1])), batch_size=32)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(test_data[0]), torch.from_numpy(test_data[1])), batch_size=32)
        return (train_loader, val_loader, test_loader)
    else:
        return (train_data, val_data, test_data)

@op_torch
class TestNbeatsPytorch(TestCase):

    def test_fit(self):
        if False:
            i = 10
            return i + 15
        (train_data, val_data, test_data) = create_data(loader=True)
        model = model_creator({'past_seq_len': 24, 'future_seq_len': 5})
        trainer = Trainer(max_epochs=1)
        pl_model = Trainer.compile(model, loss=torch.nn.MSELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=0.005))
        trainer.fit(pl_model, train_data)