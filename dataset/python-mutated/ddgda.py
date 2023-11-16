from pathlib import Path
import pickle
from typing import Optional, Union
import pandas as pd
import yaml
from qlib.contrib.meta.data_selection.dataset import InternalData, MetaDatasetDS
from qlib.contrib.meta.data_selection.model import MetaModelDS
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.meta.task import MetaTask
from qlib.model.trainer import TrainerR
from qlib.typehint import Literal
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.task.utils import replace_task_handler_with_cache
from .base import Rolling
LGBM_MODEL = '\nclass: LGBModel\nmodule_path: qlib.contrib.model.gbdt\nkwargs:\n    loss: mse\n    colsample_bytree: 0.8879\n    learning_rate: 0.2\n    subsample: 0.8789\n    lambda_l1: 205.6999\n    lambda_l2: 580.9768\n    max_depth: 8\n    num_leaves: 210\n    num_threads: 20\n'
LGBM_MODEL = yaml.load(LGBM_MODEL, Loader=yaml.FullLoader)
LINEAR_MODEL = '\nclass: LinearModel\nmodule_path: qlib.contrib.model.linear\nkwargs:\n    estimator: ridge\n    alpha: 0.05\n'
LINEAR_MODEL = yaml.load(LINEAR_MODEL, Loader=yaml.FullLoader)
PROC_ARGS = '\ninfer_processors:\n    - class: RobustZScoreNorm\n      kwargs:\n          fields_group: feature\n          clip_outlier: true\n    - class: Fillna\n      kwargs:\n          fields_group: feature\nlearn_processors:\n    - class: DropnaLabel\n    - class: CSRankNorm\n      kwargs:\n          fields_group: label\n'
PROC_ARGS = yaml.load(PROC_ARGS, Loader=yaml.FullLoader)
UTIL_MODEL_TYPE = Literal['linear', 'gbdt']

class DDGDA(Rolling):
    """
    It is a rolling based on DDG-DA

    **NOTE**
    before running the example, please clean your previous results with following command
    - `rm -r mlruns`
    """

    def __init__(self, sim_task_model: UTIL_MODEL_TYPE='gbdt', meta_1st_train_end: Optional[str]=None, alpha: float=0.01, working_dir: Optional[Union[str, Path]]=None, **kwargs):
        if False:
            return 10
        '\n\n        Parameters\n        ----------\n        sim_task_model: Literal["linear", "gbdt"] = "gbdt",\n            The model for calculating similarity between data.\n        meta_1st_train_end: Optional[str]\n            the datetime of training end of the first meta_task\n        alpha: float\n            Setting the L2 regularization for ridge\n            The `alpha` is only passed to MetaModelDS (it is not passed to sim_task_model currently..)\n        '
        self.meta_exp_name = 'DDG-DA'
        self.sim_task_model: UTIL_MODEL_TYPE = sim_task_model
        self.alpha = alpha
        self.meta_1st_train_end = meta_1st_train_end
        super().__init__(**kwargs)
        self.working_dir = self.conf_path.parent if working_dir is None else Path(working_dir)
        self.proxy_hd = self.working_dir / 'handler_proxy.pkl'

    def _adjust_task(self, task: dict, astype: UTIL_MODEL_TYPE):
        if False:
            while True:
                i = 10
        '\n        some task are use for special purpose.\n        For example:\n        - GBDT for calculating feature importance\n        - Linear or GBDT for calculating similarity\n        - Datset (well processed) that aligned to Linear that for meta learning\n        '
        handler = task['dataset'].setdefault('kwargs', {}).setdefault('handler', {})
        if astype == 'gbdt':
            task['model'] = LGBM_MODEL
            if isinstance(handler, dict):
                for k in ['infer_processors', 'learn_processors']:
                    if k in handler.setdefault('kwargs', {}):
                        handler['kwargs'].pop(k)
        elif astype == 'linear':
            task['model'] = LINEAR_MODEL
            handler['kwargs'].update(PROC_ARGS)
        else:
            raise ValueError(f'astype not supported: {astype}')
        return task

    def _get_feature_importance(self):
        if False:
            while True:
                i = 10
        task = self.basic_task(enable_handler_cache=False)
        task = self._adjust_task(task, astype='gbdt')
        task = replace_task_handler_with_cache(task, self.working_dir)
        with R.start(experiment_name='feature_importance'):
            model = init_instance_by_config(task['model'])
            dataset = init_instance_by_config(task['dataset'])
            model.fit(dataset)
        fi = model.get_feature_importance()
        df = dataset.prepare(segments=slice(None), col_set='feature', data_key=DataHandlerLP.DK_R)
        cols = df.columns
        fi_named = {cols[int(k.split('_')[1])]: imp for (k, imp) in fi.to_dict().items()}
        return pd.Series(fi_named)

    def _dump_data_for_proxy_model(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Dump data for training meta model.\n        The meta model will be trained upon the proxy forecasting model.\n        This dataset is for the proxy forecasting model.\n        '
        topk = 30
        fi = self._get_feature_importance()
        col_selected = fi.nlargest(topk)
        task = self._adjust_task(self.basic_task(enable_handler_cache=False), self.sim_task_model)
        task = replace_task_handler_with_cache(task, self.working_dir)
        dataset = init_instance_by_config(task['dataset'])
        prep_ds = dataset.prepare(slice(None), col_set=['feature', 'label'], data_key=DataHandlerLP.DK_L)
        feature_df = prep_ds['feature']
        label_df = prep_ds['label']
        feature_selected = feature_df.loc[:, col_selected.index]
        feature_selected = feature_selected.groupby('datetime', group_keys=False).apply(lambda df: (df - df.mean()).div(df.std()))
        feature_selected = feature_selected.fillna(0.0)
        df_all = {'label': label_df.reindex(feature_selected.index), 'feature': feature_selected}
        df_all = pd.concat(df_all, axis=1)
        df_all.to_pickle(self.working_dir / 'fea_label_df.pkl')
        handler = DataHandlerLP(data_loader={'class': 'qlib.data.dataset.loader.StaticDataLoader', 'kwargs': {'config': self.working_dir / 'fea_label_df.pkl'}})
        handler.to_pickle(self.working_dir / self.proxy_hd, dump_all=True)

    @property
    def _internal_data_path(self):
        if False:
            i = 10
            return i + 15
        return self.working_dir / f'internal_data_s{self.step}.pkl'

    def _dump_meta_ipt(self):
        if False:
            i = 10
            return i + 15
        '\n        Dump data for training meta model.\n        This function will dump the input data for meta model\n        '
        sim_task = self._adjust_task(self.basic_task(enable_handler_cache=False), astype=self.sim_task_model)
        sim_task = replace_task_handler_with_cache(sim_task, self.working_dir)
        if self.sim_task_model == 'gbdt':
            sim_task['model'].setdefault('kwargs', {}).update({'early_stopping_rounds': None, 'num_boost_round': 150})
        exp_name_sim = f'data_sim_s{self.step}'
        internal_data = InternalData(sim_task, self.step, exp_name=exp_name_sim)
        internal_data.setup(trainer=TrainerR)
        with self._internal_data_path.open('wb') as f:
            pickle.dump(internal_data, f)

    def _train_meta_model(self, fill_method='max'):
        if False:
            for i in range(10):
                print('nop')
        '\n        training a meta model based on a simplified linear proxy model;\n        '
        train_start = '2008-01-01' if self.train_start is None else self.train_start
        train_end = '2010-12-31' if self.meta_1st_train_end is None else self.meta_1st_train_end
        test_start = (pd.Timestamp(train_end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        proxy_forecast_model_task = {'dataset': {'class': 'qlib.data.dataset.DatasetH', 'kwargs': {'handler': f'file://{(self.working_dir / self.proxy_hd).absolute()}', 'segments': {'train': (train_start, train_end), 'test': (test_start, self.basic_task()['dataset']['kwargs']['segments']['test'][1])}}}}
        kwargs = dict(task_tpl=proxy_forecast_model_task, step=self.step, segments=0.62, trunc_days=1 + self.horizon, hist_step_n=30, fill_method=fill_method, rolling_ext_days=0)
        with self._internal_data_path.open('rb') as f:
            internal_data = pickle.load(f)
        md = MetaDatasetDS(exp_name=internal_data, **kwargs)
        with R.start(experiment_name=self.meta_exp_name):
            R.log_params(**kwargs)
            mm = MetaModelDS(step=self.step, hist_step_n=kwargs['hist_step_n'], lr=0.001, max_epoch=30, seed=43, alpha=self.alpha)
            mm.fit(md)
            R.save_objects(model=mm)

    @property
    def _task_path(self):
        if False:
            print('Hello World!')
        return self.working_dir / f'tasks_s{self.step}.pkl'

    def get_task_list(self):
        if False:
            print('Hello World!')
        '\n        Leverage meta-model for inference:\n        - Given\n            - baseline tasks\n            - input for meta model(internal data)\n            - meta model (its learnt knowledge on proxy forecasting model is expected to transfer to normal forecasting model)\n        '
        exp = R.get_exp(experiment_name=self.meta_exp_name)
        rec = exp.list_recorders(rtype=exp.RT_L)[0]
        meta_model: MetaModelDS = rec.load_object('model')
        param = rec.list_params()
        trunc_days = int(param['trunc_days'])
        step = int(param['step'])
        hist_step_n = int(param['hist_step_n'])
        fill_method = param.get('fill_method', 'max')
        task_l = super().get_task_list()
        kwargs = dict(task_tpl=task_l, step=step, segments=0.0, trunc_days=trunc_days, hist_step_n=hist_step_n, fill_method=fill_method, task_mode=MetaTask.PROC_MODE_TRANSFER)
        with self._internal_data_path.open('rb') as f:
            internal_data = pickle.load(f)
        mds = MetaDatasetDS(exp_name=internal_data, **kwargs)
        new_tasks = meta_model.inference(mds)
        with self._task_path.open('wb') as f:
            pickle.dump(new_tasks, f)
        return new_tasks

    def run(self):
        if False:
            return 10
        self._dump_data_for_proxy_model()
        self._dump_meta_ipt()
        self._train_meta_model()
        super().run()