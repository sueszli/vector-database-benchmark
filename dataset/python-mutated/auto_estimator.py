from bigdl.orca.automl.search import SearchEngineFactory
from bigdl.dllib.utils.log4Error import *
from bigdl.dllib.utils.log4Error import invalidInputError
from numpy import ndarray
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union, Any
if TYPE_CHECKING:
    from bigdl.orca.automl.model.base_pytorch_model import ModelBuilder
    from pyspark.sql import DataFrame
    from ray.tune.sample import Categorical, Float, Integer, Function

class AutoEstimator:
    """
    Example:
        >>> auto_est = AutoEstimator.from_torch(model_creator=model_creator,
                                                optimizer=get_optimizer,
                                                loss=nn.BCELoss(),
                                                logs_dir="/tmp/zoo_automl_logs",
                                                resources_per_trial={"cpu": 2},
                                                name="test_fit")
        >>> auto_est.fit(data=data,
                         validation_data=validation_data,
                         search_space=create_linear_search_space(),
                         n_sampling=4,
                         epochs=1,
                         metric="accuracy")
        >>> best_model = auto_est.get_best_model()
    """

    def __init__(self, model_builder: 'ModelBuilder', logs_dir: str='/tmp/auto_estimator_logs', resources_per_trial: Optional[Dict[str, int]]=None, remote_dir: Optional[str]=None, name: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.model_builder = model_builder
        self.searcher = SearchEngineFactory.create_engine(backend='ray', logs_dir=logs_dir, resources_per_trial=resources_per_trial, remote_dir=remote_dir, name=name)
        self._fitted = False
        self.best_trial = None

    @staticmethod
    def from_torch(*, model_creator: Callable, optimizer: Callable, loss: Callable, logs_dir: str='/tmp/auto_estimator_logs', resources_per_trial: Optional[Dict[str, int]]=None, name: str='auto_pytorch_estimator', remote_dir: Optional[str]=None) -> 'AutoEstimator':
        if False:
            print('Hello World!')
        '\n        Create an AutoEstimator for torch.\n\n        :param model_creator: PyTorch model creator function.\n        :param optimizer: PyTorch optimizer creator function or pytorch optimizer name (string).\n            Note that you should specify learning rate search space with key as "lr" or LR_NAME\n            (from bigdl.orca.automl.pytorch_utils import LR_NAME) if input optimizer name.\n            Without learning rate search space specified, the default learning rate value of 1e-3\n            will be used for all estimators.\n        :param loss: PyTorch loss instance or PyTorch loss creator function\n            or pytorch loss name (string).\n        :param logs_dir: Local directory to save logs and results. It defaults to\n            "/tmp/auto_estimator_logs"\n        :param resources_per_trial: Dict. resources for each trial. e.g. {"cpu": 2}.\n        :param name: Name of the auto estimator. It defaults to "auto_pytorch_estimator"\n        :param remote_dir: String. Remote directory to sync training results and checkpoints. It\n            defaults to None and doesn\'t take effects while running in local. While running in\n            cluster, it defaults to "hdfs:///tmp/{name}".\n\n        :return: an AutoEstimator object.\n        '
        from bigdl.orca.automl.model.base_pytorch_model import PytorchModelBuilder
        model_builder = PytorchModelBuilder(model_creator=model_creator, optimizer_creator=optimizer, loss_creator=loss)
        return AutoEstimator(model_builder=model_builder, logs_dir=logs_dir, resources_per_trial=resources_per_trial, remote_dir=remote_dir, name=name)

    @staticmethod
    def from_keras(*, model_creator: Callable, logs_dir: str='/tmp/auto_estimator_logs', resources_per_trial: Optional[Dict[str, int]]=None, name: str='auto_keras_estimator', remote_dir: Optional[str]=None) -> 'AutoEstimator':
        if False:
            i = 10
            return i + 15
        '\n        Create an AutoEstimator for tensorflow keras.\n\n        :param model_creator: Tensorflow keras model creator function.\n        :param logs_dir: Local directory to save logs and results. It defaults to\n            "/tmp/auto_estimator_logs"\n        :param resources_per_trial: Dict. resources for each trial. e.g. {"cpu": 2}.\n        :param name: Name of the auto estimator. It defaults to "auto_keras_estimator"\n        :param remote_dir: String. Remote directory to sync training results and checkpoints. It\n            defaults to None and doesn\'t take effects while running in local. While running in\n            cluster, it defaults to "hdfs:///tmp/{name}".\n\n        :return: an AutoEstimator object.\n        '
        from bigdl.orca.automl.model.base_keras_model import KerasModelBuilder
        model_builder = KerasModelBuilder(model_creator=model_creator)
        return AutoEstimator(model_builder=model_builder, logs_dir=logs_dir, resources_per_trial=resources_per_trial, remote_dir=remote_dir, name=name)

    def fit(self, data: Union[Callable, Tuple['ndarray', 'ndarray'], 'DataFrame'], epochs: int=1, validation_data: Optional[Union[Callable, Tuple['ndarray', 'ndarray'], 'DataFrame']]=None, metric: Optional[Union[Callable, str]]=None, metric_mode: Optional[str]=None, metric_threshold: Optional[Union['Function', 'float', 'int']]=None, n_sampling: int=1, search_space: Optional[Dict]=None, search_alg: Optional[str]=None, search_alg_params: Optional[Dict]=None, scheduler: Optional[str]=None, scheduler_params: Optional[Dict]=None, feature_cols: Optional[List[str]]=None, label_cols: Optional[List[str]]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Automatically fit the model and search for the best hyperparameters.\n\n        :param data: train data.\n            If the AutoEstimator is created with from_torch, data can be a tuple of\n            ndarrays or a PyTorch DataLoader or a function that takes a config dictionary as\n            parameter and returns a PyTorch DataLoader.\n            If the AutoEstimator is created with from_keras, data can be a tuple of\n            ndarrays or a function that takes a config dictionary as\n            parameter and returns a Tensorflow Dataset.\n            If data is a tuple of ndarrays, it should be in the form of (x, y),\n            where x is training input data and y is training target data.\n        :param epochs: Max number of epochs to train in each trial. Defaults to 1.\n            If you have also set metric_threshold, a trial will stop if either it has been\n            optimized to the metric_threshold or it has been trained for {epochs} epochs.\n        :param validation_data: Validation data. Validation data type should be the same as data.\n        :param metric: String or customized evaluation metric function.\n            If string, metric is the evaluation metric name to optimize, e.g. "mse".\n            If callable function, it signature should be func(y_true, y_pred), where y_true and\n            y_pred are numpy ndarray. The function should return a float value as evaluation result.\n        :param metric_mode: One of ["min", "max"]. "max" means greater metric value is better.\n            You have to specify metric_mode if you use a customized metric function.\n            You don\'t have to specify metric_mode if you use the built-in metric in\n            bigdl.orca.automl.metrics.Evaluator.\n        :param metric_threshold: a trial will be terminated when metric threshold is met\n        :param n_sampling: Number of times to sample from the search_space. Defaults to 1.\n            If hp.grid_search is in search_space, the grid will be repeated n_sampling of times.\n            If this is -1, (virtually) infinite samples are generated\n            until a stopping condition is met.\n        :param search_space: a dict for search space\n        :param search_alg: str, all supported searcher provided by ray tune\n               (i.e."variant_generator", "random", "ax", "dragonfly", "skopt",\n               "hyperopt", "bayesopt", "bohb", "nevergrad", "optuna", "zoopt" and\n               "sigopt")\n        :param search_alg_params: extra parameters for searcher algorithm besides search_space,\n            metric and searcher mode\n        :param scheduler: str, all supported scheduler provided by ray tune\n        :param scheduler_params: parameters for scheduler\n        :param feature_cols: feature column names if data is Spark DataFrame.\n        :param label_cols: target column names if data is Spark DataFrame.\n        '
        if self._fitted:
            invalidInputError(False, 'This AutoEstimator has already been fitted and cannot fit again.')
        metric_mode = AutoEstimator._validate_metric_mode(metric, metric_mode)
        (feature_cols, label_cols) = AutoEstimator._check_spark_dataframe_input(data, validation_data, feature_cols, label_cols)
        self.searcher.compile(data=data, model_builder=self.model_builder, epochs=epochs, validation_data=validation_data, metric=metric, metric_mode=metric_mode, metric_threshold=metric_threshold, n_sampling=n_sampling, search_space=search_space, search_alg=search_alg, search_alg_params=search_alg_params, scheduler=scheduler, scheduler_params=scheduler_params, feature_cols=feature_cols, label_cols=label_cols)
        self.searcher.run()
        self._fitted = True

    def get_best_model(self):
        if False:
            while True:
                i = 10
        '\n        Return the best model found by the AutoEstimator\n\n        :return: the best model instance\n        '
        if not self.best_trial:
            self.best_trial = self.searcher.get_best_trial()
        best_model_path = self.best_trial.model_path
        best_config = self.best_trial.config
        best_automl_model = self.model_builder.build(best_config)
        best_automl_model.restore(best_model_path)
        return best_automl_model.model

    def get_best_config(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the best config found by the AutoEstimator\n\n        :return: A dictionary of best hyper parameters\n        '
        if not self.best_trial:
            self.best_trial = self.searcher.get_best_trial()
        best_config = self.best_trial.config
        return best_config

    def _get_best_automl_model(self):
        if False:
            return 10
        '\n        This is for internal use only.\n        Return the best automl model found by the AutoEstimator\n\n        :return: an automl base model instance\n        '
        if not self.best_trial:
            self.best_trial = self.searcher.get_best_trial()
        best_model_path = self.best_trial.model_path
        best_config = self.best_trial.config
        best_automl_model = self.model_builder.build(best_config)
        best_automl_model.restore(best_model_path)
        return best_automl_model

    @staticmethod
    def _validate_metric_mode(metric: Optional[Union[Callable, str]], mode: Optional[str]) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        if not mode:
            if callable(metric):
                invalidInputError(False, 'You must specify `metric_mode` for your metric function')
            try:
                from bigdl.orca.automl.metrics import Evaluator
                mode = Evaluator.get_metric_mode(metric)
            except ValueError:
                pass
            if not mode:
                invalidInputError(False, f'We cannot infer metric mode with metric name of {metric}. Please specify the `metric_mode` parameter in AutoEstimator.fit().')
        if mode not in ['min', 'max']:
            invalidInputError(False, "`mode` has to be one of ['min', 'max']")
        return mode

    @staticmethod
    def _check_spark_dataframe_input(data: Union[Tuple['ndarray', 'ndarray'], Callable, 'DataFrame'], validation_data: Optional[Union[Callable, Tuple['ndarray', 'ndarray'], 'DataFrame']], feature_cols: Optional[List[str]], label_cols: Optional[List[str]]) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        if False:
            return 10

        def check_cols(cols, cols_name):
            if False:
                while True:
                    i = 10
            if not cols:
                invalidInputError(False, f'You must input valid {cols_name} for Spark DataFrame data input')
            if isinstance(cols, list):
                return cols
            if not isinstance(cols, str):
                invalidInputError(False, f'{cols_name} should be a string or a list of strings, but got {type(cols)}')
            return [cols]
        from pyspark.sql import DataFrame
        if isinstance(data, DataFrame):
            feature_cols = check_cols(feature_cols, cols_name='feature_cols')
            label_cols = check_cols(label_cols, cols_name='label_cols')
            if validation_data:
                if not isinstance(validation_data, DataFrame):
                    invalidInputError(False, f'data and validation_data should be both Spark DataFrame, but got validation_data of type {type(data)}')
        return (feature_cols, label_cols)