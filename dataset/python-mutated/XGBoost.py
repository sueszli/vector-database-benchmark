from bigdl.orca.common import SafePickle
import pandas as pd
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier
from bigdl.orca.automl.metrics import Evaluator
from bigdl.orca.automl.model.abstract import BaseModel, ModelBuilder
from bigdl.dllib.utils.log4Error import invalidInputError
import logging
logger = logging.getLogger(__name__)
XGB_METRIC_NAME = {'rmse', 'rmsle', 'mae', 'mape', 'mphe', 'logloss', 'error', 'error@t', 'merror', 'mlogloss', 'auc', 'aucpr', 'ndcg', 'map', 'ndcg@n', 'map@n', 'ndcg-', 'map-', 'ndcg@n-', 'map@n-', 'poisson-nloglik', 'gamma-nloglik', 'cox-nloglik', 'gamma-deviance', 'tweedie-nloglik', 'aft-nloglik', 'interval-regression-accuracy'}

class XGBoost(BaseModel):

    def __init__(self, model_type='regressor', config=None):
        if False:
            return 10
        '\n        Initialize hyper parameters\n        :param check_optional_config:\n        :param future_seq_len:\n        '
        if not config:
            config = {}
        valid_model_type = ('regressor', 'classifier')
        if model_type not in valid_model_type:
            invalidInputError(False, f'model_type must be between {valid_model_type}. Got {model_type}')
        self.model_type = model_type
        self.n_estimators = config.get('n_estimators', 1000)
        self.max_depth = config.get('max_depth', 5)
        self.tree_method = config.get('tree_method', 'hist')
        self.n_jobs = config.get('n_jobs', -1)
        self.random_state = config.get('random_state', 2)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.min_child_weight = config.get('min_child_weight', 1)
        self.seed = config.get('seed', 0)
        self.subsample = config.get('subsample', 0.8)
        self.colsample_bytree = config.get('colsample_bytree', 0.8)
        self.gamma = config.get('gamma', 0)
        self.reg_alpha = config.get('reg_alpha', 0)
        self.reg_lambda = config.get('reg_lambda', 1)
        self.verbosity = config.get('verbosity', 0)
        self.metric = config.get('metric')
        self.model = None
        self.model_init = False
        self.config = config

    def set_params(self, **config):
        if False:
            for i in range(10):
                print('nop')
        self.n_estimators = config.get('n_estimators', self.n_estimators)
        self.max_depth = config.get('max_depth', self.max_depth)
        self.tree_method = config.get('tree_method', self.tree_method)
        self.n_jobs = config.get('n_jobs', self.n_jobs)
        self.random_state = config.get('random_state', self.random_state)
        self.learning_rate = config.get('learning_rate', self.learning_rate)
        self.min_child_weight = config.get('min_child_weight', self.min_child_weight)
        self.seed = config.get('seed', self.seed)
        self.subsample = config.get('subsample', self.subsample)
        self.colsample_bytree = config.get('colsample_bytree', self.colsample_bytree)
        self.gamma = config.get('gamma', self.gamma)
        self.reg_alpha = config.get('reg_alpha', self.reg_alpha)
        self.reg_lambda = config.get('reg_lambda', self.reg_lambda)
        self.verbosity = config.get('verbosity', self.verbosity)
        self.config.update(config)

    def _build(self, **config):
        if False:
            while True:
                i = 10
        '\n        build the models and initialize.\n        :param config: hyper parameters for building the model\n        :return:\n        '
        self.set_params(**config)
        if self.model_type == 'regressor':
            self.model = XGBRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, n_jobs=self.n_jobs, tree_method=self.tree_method, random_state=self.random_state, learning_rate=self.learning_rate, min_child_weight=self.min_child_weight, seed=self.seed, subsample=self.subsample, colsample_bytree=self.colsample_bytree, gamma=self.gamma, reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda, verbosity=self.verbosity)
        elif self.model_type == 'classifier':
            self.model = XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, n_jobs=self.n_jobs, tree_method=self.tree_method, random_state=self.random_state, learning_rate=self.learning_rate, min_child_weight=self.min_child_weight, seed=self.seed, subsample=self.subsample, colsample_bytree=self.colsample_bytree, gamma=self.gamma, reg_alpha=self.reg_alpha, objective='binary:logistic', reg_lambda=self.reg_lambda, verbosity=self.verbosity)
        else:
            invalidInputError(False, 'model_type can only be "regressor" or "classifier"')
        self.model_init = True

    def fit_eval(self, data, validation_data=None, metric=None, metric_func=None, **config):
        if False:
            print('Hello World!')
        '\n        Fit on the training data from scratch.\n        Since the rolling process is very customized in this model,\n        we enclose the rolling process inside this method.\n        :param verbose:\n        :return: the evaluation metric value\n        '
        if not self.model_init:
            self._build(**config)
        data = self._validate_data(data, 'data')
        (x, y) = (data[0], data[1])
        if validation_data is not None:
            if isinstance(validation_data, list):
                validation_data = validation_data[0]
            validation_data = self._validate_data(validation_data, 'validation_data')
            eval_set = [validation_data]
        else:
            eval_set = None
        valid_metric_names = XGB_METRIC_NAME | Evaluator.metrics_func.keys()
        default_metric = 'rmse' if self.model_type == 'regressor' else 'logloss'
        if not metric and metric_func:
            metric_name = metric_func.__name__
        else:
            metric_name = metric or self.metric or default_metric
        if not metric_func and metric_name not in valid_metric_names:
            invalidInputError(False, f'Got invalid metric name of {metric_name} for XGBoost. Valid metrics are {valid_metric_names}')
        if metric_name in XGB_METRIC_NAME and (not metric_func):
            self.model.fit(x, y, eval_set=eval_set, eval_metric=metric_name)
            vals = self.model.evals_result_.get('validation_0').get(metric_name)
            return {metric_name: vals[-1]}
        else:
            self.model.fit(x, y, eval_set=eval_set, eval_metric=default_metric)
            eval_result = self.evaluate(validation_data[0], validation_data[1], metrics=[metric_func or metric_name])[0]
            return {metric_name: eval_result}

    def _validate_data(self, data, name):
        if False:
            i = 10
            return i + 15
        if callable(data):
            data = data(self.config)
            if not isinstance(data, tuple) or isinstance(data, list):
                invalidInputError(False, f'You must input a data create function which returns a tuple or a list of (x, y) for {name} in XGBoost. Your function returns a {data.__class__.__name__} instead')
            if len(data) != 2:
                invalidInputError(False, f'You must input a data create function which returns a tuple or a list containing two elements of (x, y) for {name} in XGBoost. Your data create function returns {len(data)} elements instead')
        if not (isinstance(data, tuple) or isinstance(data, list)):
            invalidInputError(False, f'You must input a tuple or a list of (x, y) for {name} in XGBoost.Got {data.__class__.__name__}')
        if len(data) != 2:
            invalidInputError(False, f'You must input a tuple or a list containing two elements of (x, y).Got {len(data)} elements for {name} in XGBoost')
        return data

    def predict(self, x):
        if False:
            print('Hello World!')
        "\n        Predict horizon time-points ahead the input x in fit_eval\n        :param x: We don't support input x currently.\n        :param horizon: horizon length to predict\n        :param mc:\n        :return:\n        "
        if x is None:
            invalidInputError(False, 'Input invalid x of None')
        if self.model is None:
            invalidInputError(False, 'Needs to call fit_eval or restore first before calling predict')
        self.model.n_jobs = self.n_jobs
        out = self.model.predict(x)
        return out

    def evaluate(self, x, y, metrics=['mse']):
        if False:
            for i in range(10):
                print('nop')
        "\n        Evaluate on the prediction results and y. We predict horizon time-points ahead the input x\n        in fit_eval before evaluation, where the horizon length equals the second dimension size of\n        y.\n        :param x: We don't support input x currently.\n        :param y: target. We interpret the second dimension of y as the horizon length for\n            evaluation.\n        :param metrics: a list of metrics in string format\n        :return: a list of metric evaluation results\n        "
        if x is None:
            invalidInputError(False, 'Input invalid x of None')
        if y is None:
            invalidInputError(False, 'Input invalid y of None')
        if self.model is None:
            invalidInputError(False, 'Needs to call fit_eval or restore first before calling predict')
        if isinstance(y, pd.DataFrame):
            y = y.values
        self.model.n_jobs = self.n_jobs
        y_pred = self.predict(x)
        result_list = []
        for metric in metrics:
            if callable(metric):
                result_list.append(metric(y, y_pred))
            else:
                result_list.append(Evaluator.evaluate(metric, y, y_pred))
        return result_list

    def save(self, checkpoint):
        if False:
            while True:
                i = 10
        SafePickle.dump(self.model, open(checkpoint, 'wb'))

    def restore(self, checkpoint):
        if False:
            for i in range(10):
                print('nop')
        with open(checkpoint, 'rb') as f:
            self.model = SafePickle.load(f)
        self.model_init = True

    def _get_required_parameters(self):
        if False:
            i = 10
            return i + 15
        return {}

    def _get_optional_parameters(self):
        if False:
            while True:
                i = 10
        param = self.model.get_xgb_params
        return param

class XGBoostModelBuilder(ModelBuilder):

    def __init__(self, model_type='regressor', cpus_per_trial=1, **xgb_configs):
        if False:
            print('Hello World!')
        self.model_type = model_type
        self.model_config = xgb_configs.copy()
        if 'n_jobs' in xgb_configs and xgb_configs['n_jobs'] != cpus_per_trial:
            logger.warning(f"Found n_jobs={xgb_configs['n_jobs']} in xgb_configs. It will not take effect since we assign cpus_per_trials(={cpus_per_trial}) to xgboost n_jobs. Please throw an issue if you do need different values for xgboost n_jobs and cpus_per_trials.")
        self.model_config['n_jobs'] = cpus_per_trial

    def build(self, config):
        if False:
            while True:
                i = 10
        model = XGBoost(model_type=self.model_type, config=self.model_config)
        model._build(**config)
        return model