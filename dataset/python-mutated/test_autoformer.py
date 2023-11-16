from unittest import TestCase
from bigdl.chronos.utils import LazyImport
model_creator = LazyImport('bigdl.chronos.model.autoformer.model_creator')
TSTrainer = LazyImport('bigdl.chronos.pytorch.TSTrainer')
torch = LazyImport('torch')
TensorDataset = LazyImport('torch.utils.data.TensorDataset')
DataLoader = LazyImport('torch.utils.data.DataLoader')
from bigdl.chronos.data import TSDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tempfile
import os
from .. import op_torch

def get_ts_df():
    if False:
        for i in range(10):
            print('nop')
    sample_num = np.random.randint(400, 500)
    train_df = pd.DataFrame({'datetime': pd.date_range('1/1/2019', periods=sample_num), 'value': np.random.randn(sample_num)})
    return train_df

def get_tsdata(mode='train'):
    if False:
        i = 10
        return i + 15
    df = get_ts_df()
    tsdata = TSDataset.from_pandas(df, dt_col='datetime', target_col='value')
    stand = StandardScaler()
    tsdata.impute('last').scale(stand, fit=mode == 'train').roll(lookback=48, horizon=12, time_enc=True, label_len=12)
    return tsdata

@op_torch
class TestAutoformerPytorch(TestCase):

    def test_fit(self):
        if False:
            while True:
                i = 10
        tsdata = get_tsdata()
        data = tsdata.to_numpy()
        dataloader = DataLoader(TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(data[1]), torch.from_numpy(data[2]), torch.from_numpy(data[3])), batch_size=32)
        config_dict = {'seq_len': 48, 'label_len': 12, 'pred_len': 12, 'enc_in': 1, 'dec_in': 1, 'c_out': 1, 'freq': 'd'}
        model = model_creator(config_dict)
        trainer = TSTrainer(max_epochs=2)
        trainer.fit(model, dataloader)