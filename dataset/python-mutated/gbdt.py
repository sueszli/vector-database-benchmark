import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Text, Tuple, Union
from ...model.base import ModelFT
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import LightGBMFInt
from ...data.dataset.weight import Reweighter
from qlib.workflow import R

class LGBModel(ModelFT, LightGBMFInt):
    """LightGBM Model"""

    def __init__(self, loss='mse', early_stopping_rounds=50, num_boost_round=1000, **kwargs):
        if False:
            return 10
        if loss not in {'mse', 'binary'}:
            raise NotImplementedError
        self.params = {'objective': loss, 'verbosity': -1}
        self.params.update(kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model = None

    def _prepare_data(self, dataset: DatasetH, reweighter=None) -> List[Tuple[lgb.Dataset, str]]:
        if False:
            i = 10
            return i + 15
        '\n        The motivation of current version is to make validation optional\n        - train segment is necessary;\n        '
        ds_l = []
        assert 'train' in dataset.segments
        for key in ['train', 'valid']:
            if key in dataset.segments:
                df = dataset.prepare(key, col_set=['feature', 'label'], data_key=DataHandlerLP.DK_L)
                if df.empty:
                    raise ValueError('Empty data from dataset, please check your dataset config.')
                (x, y) = (df['feature'], df['label'])
                if y.values.ndim == 2 and y.values.shape[1] == 1:
                    y = np.squeeze(y.values)
                else:
                    raise ValueError("LightGBM doesn't support multi-label training")
                if reweighter is None:
                    w = None
                elif isinstance(reweighter, Reweighter):
                    w = reweighter.reweight(df)
                else:
                    raise ValueError('Unsupported reweighter type.')
                ds_l.append((lgb.Dataset(x.values, label=y, weight=w), key))
        return ds_l

    def fit(self, dataset: DatasetH, num_boost_round=None, early_stopping_rounds=None, verbose_eval=20, evals_result=None, reweighter=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if evals_result is None:
            evals_result = {}
        ds_l = self._prepare_data(dataset, reweighter)
        (ds, names) = list(zip(*ds_l))
        early_stopping_callback = lgb.early_stopping(self.early_stopping_rounds if early_stopping_rounds is None else early_stopping_rounds)
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        evals_result_callback = lgb.record_evaluation(evals_result)
        self.model = lgb.train(self.params, ds[0], num_boost_round=self.num_boost_round if num_boost_round is None else num_boost_round, valid_sets=ds, valid_names=names, callbacks=[early_stopping_callback, verbose_eval_callback, evals_result_callback], **kwargs)
        for k in names:
            for (key, val) in evals_result[k].items():
                name = f'{key}.{k}'
                for (epoch, m) in enumerate(val):
                    R.log_metrics(**{name.replace('@', '_'): m}, step=epoch)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice]='test'):
        if False:
            print('Hello World!')
        if self.model is None:
            raise ValueError('model is not fitted yet!')
        x_test = dataset.prepare(segment, col_set='feature', data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x_test.values), index=x_test.index)

    def finetune(self, dataset: DatasetH, num_boost_round=10, verbose_eval=20, reweighter=None):
        if False:
            i = 10
            return i + 15
        '\n        finetune model\n\n        Parameters\n        ----------\n        dataset : DatasetH\n            dataset for finetuning\n        num_boost_round : int\n            number of round to finetune model\n        verbose_eval : int\n            verbose level\n        '
        (dtrain, _) = self._prepare_data(dataset, reweighter)
        if dtrain.empty:
            raise ValueError('Empty data from dataset, please check your dataset config.')
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        self.model = lgb.train(self.params, dtrain, num_boost_round=num_boost_round, init_model=self.model, valid_sets=[dtrain], valid_names=['train'], callbacks=[verbose_eval_callback])