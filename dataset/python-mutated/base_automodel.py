import os
import json
from bigdl.chronos.autots.utils import recalculate_n_sampling
import warnings

class BaseAutomodel:

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if self.backend.startswith('torch'):
            from bigdl.orca.automl.model.base_pytorch_model import PytorchModelBuilder
            self._DEFAULT_BEST_MODEL_DIR = 'best_model.ckpt'
            self._DEFAULT_BEST_CONFIG_DIR = 'best_config.json'
            model_builder = PytorchModelBuilder(model_creator=self._model_creator, optimizer_creator=self.optimizer, loss_creator=self.loss)
        elif self.backend.startswith('keras'):
            from bigdl.orca.automl.model.base_keras_model import KerasModelBuilder
            self.search_space.update({'optimizer': self.optimizer, 'loss': self.loss})
            model_builder = KerasModelBuilder(model_creator=self._model_creator)
            self._DEFAULT_BEST_MODEL_DIR = 'best_keras_model.ckpt'
            self._DEFAULT_BEST_CONFIG_DIR = 'best_keras_config.json'
        from bigdl.orca.automl.auto_estimator import AutoEstimator
        self.auto_est = AutoEstimator(model_builder, **self._auto_est_config)
        self.best_model = None

    def fit(self, data, epochs=1, batch_size=32, validation_data=None, metric_threshold=None, n_sampling=1, search_alg=None, search_alg_params=None, scheduler=None, scheduler_params=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Automatically fit the model and search for the best hyper parameters.\n\n        :param data: train data.\n               data can be a tuple of ndarrays or a PyTorch DataLoader\n               or a function that takes a config dictionary as parameter and returns a\n               PyTorch DataLoader.\n        :param epochs: Max number of epochs to train in each trial. Defaults to 1.\n               If you have also set metric_threshold, a trial will stop if either it has been\n               optimized to the metric_threshold or it has been trained for {epochs} epochs.\n        :param batch_size: Int or hp sampling function from an integer space. Training batch size.\n               It defaults to 32.\n        :param validation_data: Validation data. Validation data type should be the same as data.\n        :param metric_threshold: a trial will be terminated when metric threshold is met.\n        :param n_sampling: Number of trials to evaluate in total. Defaults to 1.\n               If hp.grid_search is in search_space, the grid will be run n_sampling of trials\n               and round up n_sampling according to hp.grid_search.\n               If this is -1, (virtually) infinite samples are generated\n               until a stopping condition is met.\n        :param search_alg: str, all supported searcher provided by ray tune\n               (i.e."variant_generator", "random", "ax", "dragonfly", "skopt",\n               "hyperopt", "bayesopt", "bohb", "nevergrad", "optuna", "zoopt" and\n               "sigopt").\n        :param search_alg_params: extra parameters for searcher algorithm besides search_space,\n               metric and searcher mode.\n        :param scheduler: str, all supported scheduler provided by ray tune.\n        :param scheduler_params: parameters for scheduler.\n        '
        self.search_space['batch_size'] = batch_size
        n_sampling = recalculate_n_sampling(self.search_space, n_sampling) if n_sampling != -1 else -1
        self.auto_est.fit(data=data, epochs=epochs, validation_data=validation_data, metric=self.metric, metric_mode=self.metric_mode, metric_threshold=metric_threshold, n_sampling=n_sampling, search_space=self.search_space, search_alg=search_alg, search_alg_params=search_alg_params, scheduler=scheduler, scheduler_params=scheduler_params)
        self.best_model = self.auto_est._get_best_automl_model()
        self.best_config = self.auto_est.get_best_config()

    def predict(self, data, batch_size=32):
        if False:
            print('Hello World!')
        "\n        Predict using a the trained model after HPO(Hyper Parameter Optimization).\n\n        :param data: a numpy ndarray x, where x's shape is (num_samples, lookback, feature_dim)\n               where lookback and feature_dim should be the same as past_seq_len and\n               input_feature_num.\n        :param batch_size: predict batch size. The value will not affect predict\n               result but will affect resources cost(e.g. memory and time). The value\n               defaults to 32.\n\n        :return: A numpy array with shape (num_samples, horizon, target_dim).\n        "
        from bigdl.nano.utils.common import invalidInputError
        if self.best_model is None:
            invalidInputError(False, 'You must call fit or load first before calling predict!')
        return self.best_model.predict(data, batch_size=batch_size)

    def predict_with_onnx(self, data, batch_size=32, dirname=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Predict using a the trained model after HPO(Hyper Parameter Optimization).\n\n        Be sure to install onnx and onnxruntime to enable this function. The method\n        will give exactly the same result as .predict() but with higher throughput\n        and lower latency. keras will support onnx later.\n\n        :param data: a numpy ndarray x, where x's shape is (num_samples, lookback, feature_dim)\n               where lookback and feature_dim should be the same as past_seq_len and\n               input_feature_num.\n        :param batch_size: predict batch size. The value will not affect predict\n               result but will affect resources cost(e.g. memory and time). The value\n               defaults to 32.\n        :param dirname: The directory to save onnx model file. This value defaults\n               to None for no saving file.\n\n        :return: A numpy array with shape (num_samples, horizon, target_dim).\n        "
        from bigdl.nano.utils.common import invalidInputError
        if self.backend.startswith('keras'):
            invalidInputError(False, 'Currenctly, keras not support onnx method.')
        if self.best_model is None:
            invalidInputError(False, 'You must call fit or load first before calling predict!')
        return self.best_model.predict_with_onnx(data, batch_size=batch_size, dirname=dirname)

    def evaluate(self, data, batch_size=32, metrics=['mse'], multioutput='raw_values'):
        if False:
            while True:
                i = 10
        "\n        Evaluate using a the trained model after HPO(Hyper Parameter Optimization).\n\n        Please note that evaluate result is calculated by scaled y and yhat. If you scaled\n        your data (e.g. use .scale() on the TSDataset) please follow the following code\n        snap to evaluate your result if you need to evaluate on unscaled data.\n\n        >>> from bigdl.orca.automl.metrics import Evaluator\n        >>> y_hat = automodel.predict(x)\n        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods\n        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods\n        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)\n\n        :param data: a numpy ndarray tuple (x, y) x's shape is (num_samples, lookback,\n               feature_dim) where lookback and feature_dim should be the same as\n               past_seq_len and input_feature_num. y's shape is (num_samples, horizon,\n               target_dim), where horizon and target_dim should be the same as\n               future_seq_len and output_target_num.\n        :param batch_size: evaluate batch size. The value will not affect evaluate\n               result but will affect resources cost(e.g. memory and time).\n        :param metrics: list of string or callable. e.g. ['mse'] or [customized_metrics]\n               If callable function, it signature should be func(y_true, y_pred), where y_true and\n               y_pred are numpy ndarray. The function should return a float value as evaluation\n               result.\n        :param multioutput: Defines aggregating of multiple output values.\n               String in ['raw_values', 'uniform_average']. The value defaults to\n               'raw_values'.\n\n        :return: A list of evaluation results. Each item represents a metric.\n        "
        from bigdl.nano.utils.common import invalidInputError
        if self.best_model is None:
            invalidInputError(False, 'You must call fit or load first before calling predict!')
        return self.best_model.evaluate(data[0], data[1], metrics=metrics, multioutput=multioutput, batch_size=batch_size)

    def evaluate_with_onnx(self, data, batch_size=32, metrics=['mse'], dirname=None, multioutput='raw_values'):
        if False:
            i = 10
            return i + 15
        "\n        Evaluate using a the trained model after HPO(Hyper Parameter Optimization).\n\n        Be sure to install onnx and onnxruntime to enable this function. The method\n        will give exactly the same result as .evaluate() but with higher throughput\n        and lower latency. keras will support onnx later.\n\n        Please note that evaluate result is calculated by scaled y and yhat. If you scaled\n        your data (e.g. use .scale() on the TSDataset) please follow the following code\n        snap to evaluate your result if you need to evaluate on unscaled data.\n\n        >>> from bigdl.orca.automl.metrics import Evaluator\n        >>> y_hat = automodel.predict_with_onnx(x)\n        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods\n        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods\n        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)\n\n        :param data: a numpy ndarray tuple (x, y) x's shape is (num_samples, lookback,\n               feature_dim) where lookback and feature_dim should be the same as\n               past_seq_len and input_feature_num. y's shape is (num_samples, horizon,\n               target_dim), where horizon and target_dim should be the same as\n               future_seq_len and output_target_num.\n        :param batch_size: evaluate batch size. The value will not affect evaluate\n               result but will affect resources cost(e.g. memory and time).\n        :param metrics: list of string or callable. e.g. ['mse'] or [customized_metrics]\n               If callable function, it signature should be func(y_true, y_pred), where y_true and\n               y_pred are numpy ndarray. The function should return a float value as evaluation\n               result.\n        :param dirname: The directory to save onnx model file. This value defaults\n               to None for no saving file.\n        :param multioutput: Defines aggregating of multiple output values.\n               String in ['raw_values', 'uniform_average']. The value defaults to\n               'raw_values'.\n\n        :return: A list of evaluation results. Each item represents a metric.\n        "
        from bigdl.nano.utils.common import invalidInputError
        if self.backend.startswith('keras'):
            invalidInputError(False, 'Currenctly, keras not support onnx method.')
        if self.best_model is None:
            invalidInputError(False, 'You must call fit or load first before calling predict!')
        return self.best_model.evaluate_with_onnx(data[0], data[1], metrics=metrics, dirname=dirname, multioutput=multioutput, batch_size=batch_size)

    def save(self, checkpoint_path):
        if False:
            return 10
        '\n        Save the best model.\n\n        Please note that if you only want the pytorch model or onnx model\n        file, you can call .get_model() or .export_onnx_file(). The checkpoint\n        file generated by .save() method can only be used by .load() in automodel.\n        If you specify "keras" as backend, file name will be best_keras_config.json\n        and best_keras_model.ckpt.\n\n        :param checkpoint_path: The location you want to save the best model.\n        '
        from bigdl.nano.utils.common import invalidInputError
        if self.best_model is None:
            invalidInputError(False, 'You must call fit or load first before calling predict!')
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        model_path = os.path.join(checkpoint_path, self._DEFAULT_BEST_MODEL_DIR)
        best_config_path = os.path.join(checkpoint_path, self._DEFAULT_BEST_CONFIG_DIR)
        self.best_model.save(model_path)
        with open(best_config_path, 'w') as f:
            json.dump(self.best_config, f)

    def load(self, checkpoint_path):
        if False:
            i = 10
            return i + 15
        '\n        restore the best model.\n\n        :param checkpoint_path: The checkpoint location you want to load the best model.\n        '
        model_path = os.path.join(checkpoint_path, self._DEFAULT_BEST_MODEL_DIR)
        best_config_path = os.path.join(checkpoint_path, self._DEFAULT_BEST_CONFIG_DIR)
        self.best_model.restore(model_path)
        with open(best_config_path, 'r') as f:
            self.best_config = json.load(f)

    def build_onnx(self, thread_num=1, sess_options=None):
        if False:
            return 10
        '\n        Build onnx model to speed up inference and reduce latency.\n        The method is Not required to call before predict_with_onnx,\n        evaluate_with_onnx or export_onnx_file.\n        It is recommended to use when you want to:\n\n        | 1. Strictly control the thread to be used during inferencing.\n        | 2. Alleviate the cold start problem when you call predict_with_onnx\n             for the first time.\n\n        :param thread_num: int, the num of thread limit. The value is set to 1 by\n               default where no limit is set. Besides, the environment variable\n               `OMP_NUM_THREADS` is suggested to be same as `thread_num`.\n        :param sess_options: an onnxruntime.SessionOptions instance, if you set this\n               other than None, a new onnxruntime session will be built on this setting\n               and ignore other settings you assigned(e.g. thread_num...).\n\n        Example:\n            >>> # to pre build onnx sess\n            >>> automodel.build_onnx(thread_num=2)  # build onnx runtime sess for two threads\n            >>> pred = automodel.predict_with_onnx(data)\n            >>> # ------------------------------------------------------\n            >>> # directly call onnx related method is also supported\n            >>> # default to build onnx runtime sess for single thread\n            >>> pred = automodel.predict_with_onnx(data)\n        '
        from bigdl.nano.utils.common import invalidInputError
        if self.backend.startswith('keras'):
            invalidInputError(False, 'Currenctly, keras not support onnx method.')
        import onnxruntime
        if sess_options is not None and (not isinstance(sess_options, onnxruntime.SessionOptions)):
            invalidInputError(False, f'sess_options should be an onnxruntime.SessionOptions instance, but found {type(sess_options)}')
        if self.distributed:
            invalidInputError(False, 'build_onnx has not been supported for distributed forecaster. You can call .to_local() to transform the forecaster to a non-distributed version.')
        try:
            OMP_NUM_THREADS = os.getenv('OMP_NUM_THREADS')
        except KeyError:
            OMP_NUM_THREADS = 0
        if OMP_NUM_THREADS != str(thread_num):
            warnings.warn(f"The environment variable OMP_NUM_THREADS is suggested to be same as thread_num.You can use 'export OMP_NUM_THREADS={thread_num}'.")
        import torch
        dummy_input = torch.rand(1, self.best_config['past_seq_len'], self.best_config['input_feature_num'])
        self.best_model._build_onnx(dummy_input, dirname=None, thread_num=thread_num, sess_options=None)

    def export_onnx_file(self, dirname):
        if False:
            print('Hello World!')
        '\n        Save the onnx model file to the disk.\n\n        :param dirname: The dir location you want to save the onnx file.\n        '
        from bigdl.nano.utils.common import invalidInputError
        if self.backend.startswith('keras'):
            invalidInputError(False, 'Currenctly, keras not support onnx method.')
        if self.distributed:
            invalidInputError(False, 'export_onnx_file has not been supported for distributed forecaster. You can call .to_local() to transform the forecaster to a non-distributed version.')
        import torch
        dummy_input = torch.rand(1, self.best_config['past_seq_len'], self.best_config['input_feature_num'])
        self.best_model._build_onnx(dummy_input, dirname)

    def get_best_model(self):
        if False:
            print('Hello World!')
        '\n        Get the best pytorch model.\n        '
        return self.auto_est.get_best_model()

    def get_best_config(self):
        if False:
            print('Hello World!')
        '\n        Get the best configuration\n\n        :return: A dictionary of best hyper parameters\n        '
        return self.best_config

    def _get_best_automl_model(self):
        if False:
            print('Hello World!')
        return self.best_model