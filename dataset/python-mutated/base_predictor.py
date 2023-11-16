import pandas as pd
import os
from abc import abstractmethod
from bigdl.orca.automl.metrics import Evaluator
from bigdl.chronos.autots.deprecated.pipeline.time_sequence import TimeSequencePipeline
from bigdl.orca.automl.search.utils import process
from bigdl.chronos.autots.deprecated.config.recipe import *
from bigdl.orca.ray import OrcaRayContext
from bigdl.orca.automl.auto_estimator import AutoEstimator
ALLOWED_FIT_METRICS = ('mse', 'mae', 'r2')

class BasePredictor(object):

    def __init__(self, name='automl', logs_dir='~/bigdl_automl_logs', search_alg=None, search_alg_params=None, scheduler=None, scheduler_params=None):
        if False:
            for i in range(10):
                print('nop')
        self.logs_dir = logs_dir
        self.name = name
        self.search_alg = search_alg
        self.search_alg_params = search_alg_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

    @abstractmethod
    def get_model_builder(self):
        if False:
            for i in range(10):
                print('nop')
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError(False, 'get_model_builder not implement')

    def _check_df(self, df):
        if False:
            for i in range(10):
                print('nop')
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError(isinstance(df, pd.DataFrame) and df.empty is False, 'You should input a valid data frame')

    @staticmethod
    def _check_fit_metric(metric):
        if False:
            print('Hello World!')
        from bigdl.nano.utils.common import invalidInputError
        if metric not in ALLOWED_FIT_METRICS:
            invalidInputError(False, f'metric {metric} is not supported for fit. Input metric should be among {ALLOWED_FIT_METRICS}')

    def fit(self, input_df, validation_df=None, metric='mse', recipe=SmokeRecipe(), mc=False, resources_per_trial={'cpu': 2}, upload_dir=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Trains the model for time sequence prediction.\n        If future sequence length > 1, use seq2seq model, else use vanilla LSTM model.\n        :param input_df: The input time series data frame, Example:\n         datetime   value   "extra feature 1"   "extra feature 2"\n         2019-01-01 1.9 1   2\n         2019-01-02 2.3 0   2\n        :param validation_df: validation data\n        :param metric: String. Metric used for train and validation. Available values are\n                       "mean_squared_error" or "r_square"\n        :param recipe: a Recipe object. Various recipes covers different search space and stopping\n                      criteria. Default is SmokeRecipe().\n        :param resources_per_trial: Machine resources to allocate per trial,\n            e.g. ``{"cpu": 64, "gpu": 8}`\n        :param upload_dir: Optional URI to sync training results and checkpoints. We only support\n            hdfs URI for now. It defaults to\n            "hdfs:///user/{hadoop_user_name}/ray_checkpoints/{predictor_name}".\n            Where hadoop_user_name is specified in init_orca_context or init_spark_on_yarn,\n            which defaults to "root". predictor_name is the name used in predictor instantiation.\n        )\n        :return: a pipeline constructed with the best model and configs.\n        '
        self._check_df(input_df)
        if validation_df is not None:
            self._check_df(validation_df)
        ray_ctx = OrcaRayContext.get()
        is_local = ray_ctx.is_local
        if not is_local:
            if not upload_dir:
                hadoop_user_name = os.getenv('HADOOP_USER_NAME')
                upload_dir = os.path.join(os.sep, 'user', hadoop_user_name, 'ray_checkpoints', self.name)
            cmd = 'hadoop fs -mkdir -p {}'.format(upload_dir)
            process(cmd)
        else:
            upload_dir = None
        self.pipeline = self._hp_search(input_df, validation_df=validation_df, metric=metric, recipe=recipe, mc=mc, resources_per_trial=resources_per_trial, remote_dir=upload_dir)
        return self.pipeline

    def evaluate(self, input_df, metric=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Evaluate the model on a list of metrics.\n        :param input_df: The input time series data frame, Example:\n         datetime   value   "extra feature 1"   "extra feature 2"\n         2019-01-01 1.9 1   2\n         2019-01-02 2.3 0   2\n        :param metric: A list of Strings Available string values are "mean_squared_error",\n                      "r_square".\n        :return: a list of metric evaluation results.\n        '
        Evaluator.check_metric(metric)
        return self.pipeline.evaluate(input_df, metric)

    def predict(self, input_df):
        if False:
            for i in range(10):
                print('nop')
        '\n        Predict future sequence from past sequence.\n        :param input_df: The input time series data frame, Example:\n         datetime   value   "extra feature 1"   "extra feature 2"\n         2019-01-01 1.9 1   2\n         2019-01-02 2.3 0   2\n        :return: a data frame with 2 columns, the 1st is the datetime, which is the last datetime of\n            the past sequence.\n            values are the predicted future sequence values.\n            Example :\n            datetime    value_0     value_1   ...     value_2\n            2019-01-03  2           3                   9\n        '
        return self.pipeline.predict(input_df)

    def _detach_recipe(self, recipe):
        if False:
            while True:
                i = 10
        self.search_space = recipe.search_space()
        stop = recipe.runtime_params()
        self.metric_threshold = None
        if 'reward_metric' in stop.keys():
            self.mode = Evaluator.get_metric_mode(self.metric)
            self.metric_threshold = -stop['reward_metric'] if self.mode == 'min' else stop['reward_metric']
        self.epochs = stop['training_iteration']
        self.num_samples = stop['num_samples']

    def _hp_search(self, input_df, validation_df, metric, recipe, mc, resources_per_trial, remote_dir):
        if False:
            while True:
                i = 10
        model_builder = self.get_model_builder()
        self.metric = metric
        self._detach_recipe(recipe)
        auto_est = AutoEstimator(model_builder, logs_dir=self.logs_dir, resources_per_trial=resources_per_trial, name=self.name, remote_dir=remote_dir)
        auto_est.fit(data=input_df, validation_data=validation_df, search_space=self.search_space, n_sampling=self.num_samples, epochs=self.epochs, metric_threshold=self.metric_threshold, search_alg=self.search_alg, search_alg_params=self.search_alg_params, scheduler=self.scheduler, scheduler_params=self.scheduler_params, metric=metric)
        best_model = auto_est._get_best_automl_model()
        pipeline = TimeSequencePipeline(name=self.name, model=best_model)
        return pipeline