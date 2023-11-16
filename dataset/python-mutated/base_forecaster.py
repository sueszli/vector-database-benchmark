import warnings
import keras
import tensorflow as tf
import numpy as np
from bigdl.chronos.forecaster.abstract import Forecaster
from bigdl.chronos.data import TSDataset
from bigdl.chronos.metric.forecast_metrics import Evaluator
from bigdl.nano.utils.common import invalidInputError
from bigdl.chronos.forecaster.tf.utils import np_to_data_creator, tsdata_to_data_creator, np_to_xshards, xshard_to_np, np_to_tfdataset

class BaseTF2Forecaster(Forecaster):

    def __init__(self, **kwargs):
        if False:
            return 10
        if self.seed:
            warnings.warn(f"If users want to set the random seed for training, it's better to call 'tf.keras.utils.set_random_seed({self.seed})' manually.")
            from tensorflow.keras.utils import set_random_seed
            set_random_seed(seed=self.seed)
        if self.distributed:
            from bigdl.orca.learn.tf2.estimator import Estimator
            self.internal = Estimator.from_keras(model_creator=self.model_creator, config=self.model_config, workers_per_node=self.workers_per_node, backend=self.remote_distributed_backend)
        else:
            self.internal = self.model_creator({**self.model_config})
            self.accelerated_model = None
            self.accelerate_method = None
        self.fitted = False

    def fit(self, data, epochs=1, batch_size=32):
        if False:
            return 10
        "\n        Fit(Train) the forecaster.\n\n        :param data: The data support following formats:\n\n               | 1. A numpy ndarray tuple (x, y):\n               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim\n               | should be the same as past_seq_len and input_feature_num.\n               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim\n               | should be the same as future_seq_len and output_feature_num.\n               |\n               | 2. A tf.data.Dataset:\n               | A TFDataset instance which contains x and y with same shape as the tuple.\n               | x's shape is (num_samples, lookback, feature_dim),\n               | y's shape is (num_samples, horizon, target_dim).\n               | If set distributed to True, we do not recommend using tf.data.Dataset,\n               | please replace with tsdataset or numpy.ndarray.\n               |\n               | 3. A bigdl.chronos.data.tsdataset.TSDataset instance.\n               | Forecaster will automatically process the TSDataset.\n               | By default, TSDataset will be transformed to a tfdataset,\n               | Users may call `roll` on the TSDataset before calling `fit`\n               | Then the training speed will be faster but will consume more memory.\n\n        :params epochs: Number of epochs you want to train. The value defaults to 1.\n        :params batch_size: Number of batch size you want to train. The value defaults to 32.\n                Do not specify the batch_size, if your data in the form of tf.data datasets.\n        "
        if isinstance(data, TSDataset):
            if data.lookback is None:
                data.roll(lookback=self.model_config['past_seq_len'], horizon=self.model_config['future_seq_len'])
        if self.distributed:
            if isinstance(data, tuple):
                data = np_to_data_creator(data, shuffle=True)
            if isinstance(data, TSDataset):
                data = tsdata_to_data_creator(data, shuffle=True)
            from bigdl.nano.utils.common import invalidInputError
            invalidInputError(not isinstance(data, tf.data.Dataset), 'tf.data.Dataset is not supported, please replace with numpy.ndarray or TSDataset instance.')
            from bigdl.orca.common import OrcaContext
            sc = OrcaContext.get_spark_context().getConf()
            num_nodes = 1 if sc.get('spark.master').startswith('local') else int(sc.get('spark.executor.instances'))
            if batch_size % self.workers_per_node != 0:
                from bigdl.nano.utils.common import invalidInputError
                invalidInputError(False, f"Please make sure that batch_size can be divisible by the product of worker_per_node and num_nodes, but 'batch_size' is {batch_size}, 'workers_per_node' is {self.workers_per_node}, 'num_nodes' is {num_nodes}")
            batch_size //= self.workers_per_node * num_nodes
            self.internal.fit(data, epochs=epochs, batch_size=batch_size)
        elif isinstance(data, tuple):
            self.internal.fit(x=data[0], y=data[1], epochs=epochs, batch_size=batch_size)
        else:
            if isinstance(data, TSDataset):
                data = data.to_tf_dataset(shuffle=True, batch_size=batch_size)
            self.internal.fit(data, epochs=epochs)
        self.fitted = True

    def quantize(self, input_data=None, target_data=None, metric=None, conf=None, framework='tensorflow', approach='static', tuning_strategy='bayesian', relative_drop=None, absolute_drop=None, timeout=0, max_trials=1, sess_options=None, thread_num=None):
        if False:
            while True:
                i = 10
        '\n        Quantize the forecaster.\n\n        :param input_data: Input data which is used for training. Support following formats:\n\n               | 1. a numpy ndarray:\n               | The shape is (num_samples, lookback, feature_dim) where lookback and feature_dim\n               | should be the same as past_seq_len and input_feature_num.\n               |\n               | 2. TensorFlow tensor:\n               | The shape is (num_samples, lookback, feature_dim) where lookback and feature_dim\n               | should be the same as past_seq_len and input_feature_num.\n               |\n               | 3. an unbatched tf.data.Dataset. Should return a tuple of (inputs, targets).\n               |\n               | Input data will be used as calibration dataset for static quantization,\n               | as well as be used for generating input_sample to calculate latency.\n               | To avoid data leak during calibration, please use training dataset.\n\n        :param target_data: Target data. It could be either Numpy array(s) or TensorFlow tensor(s)\n               while the length should be consistent with `input_data`.\n               If `input_data` is dataset, `target_data` will be ignored.\n        :param metric: A str represent the metrics for tunning the quality of\n               quantization. You may choose from "mse", "mae", "rmse", "r2", "mape", "smape".\n        :param conf: A path to conf yaml file for quantization. Default to None,\n               using default config.\n        :param framework: A str represent the framework for quantization. You may choose from\n               "tensorflow", "onnxrt_integerops", "onnxrt_qlinearops", "openvino".\n               Default: \'tensorflow\'.\n        :param approach: str, \'static\' or \'dynamic\'. Default to \'static\'.\n               Only \'static\' approach is supported now.\n        :param tuning_strategy: str, \'bayesian\', \'basic\', \'mse\' or \'sigopt\'. Default to \'bayesian\'.\n        :param relative_drop: Float, tolerable ralative accuracy drop. Default to None,\n               e.g. set to 0.1 means that we accept a 10% increase in the metrics error.\n        :param absolute_drop: Float, tolerable ralative accuracy drop. Default to None,\n               e.g. set to 5 means that we can only accept metrics smaller than 5.\n        :param timeout: Tuning timeout (seconds). Default to 0, which means early stop.\n               Combine with max_trials field to decide when to exit.\n        :param max_trials: Max tune times. Default to 1. Combine with timeout field to\n               decide when to exit. "timeout=0, max_trials=1" means it will try quantization\n               only once and return satisfying best model.\n        :param sess_options: The session option for onnxruntime, only valid when\n               framework contains \'onnxrt_integerops\' or \'onnxrt_qlinearops\',\n               otherwise will be ignored.\n        :param thread_num: int, the num of thread limit, only valid when framework contains\n               \'onnxrt_integerops\' or \'onnxrt_qlinearops\' or \'openvino\'. The value is set to None\n               by default where no limit is set.\n        '
        from bigdl.nano.utils.common import invalidInputError
        from bigdl.nano.tf.keras import InferenceOptimizer
        if not self.quantize_available:
            invalidInputError(False, 'This model has not supported quantization.')
        if self.distributed:
            invalidInputError(False, 'quantization has not been supported for distributed forecaster. You can call .to_local() to transform the forecaster to a non-distributed version.')
        try:
            metric = _str2metric(metric)
        except Exception:
            invalidInputError(False, 'Unable to recognize the metric string you passed in.')
        accuracy_criterion = None
        if relative_drop and absolute_drop:
            invalidInputError(False, 'Please unset either `relative_drop` or `absolute_drop`.')
        if relative_drop:
            accuracy_criterion = {'relative': relative_drop, 'higher_is_better': False}
        if absolute_drop:
            accuracy_criterion = {'absolute': absolute_drop, 'higher_is_better': False}
        if '_' in framework:
            (accelerator, method) = framework.split('_')
        else:
            accelerator = framework
        if accelerator == 'tensorflow':
            accelerator = None
            method = None
        elif accelerator == 'openvino':
            method = None
        else:
            accelerator = 'onnxruntime'
            method = method[:-3]
        input_spec = tf.TensorSpec(shape=(None, self.model_config['past_seq_len'], self.model_config['input_feature_num']))
        q_model = InferenceOptimizer.quantize(self.internal, x=input_data, y=target_data, input_spec=input_spec, precision='int8', accelerator=accelerator, method=method, metric=metric, conf=conf, approach=approach, tuning_strategy=tuning_strategy, accuracy_criterion=accuracy_criterion, timeout=timeout, max_trials=max_trials, onnxruntime_session_options=sess_options, thread_num=thread_num)
        if accelerator == 'onnxruntime':
            self.accelerated_model = q_model
            self.accelerate_method = 'onnxruntime_int8'
        if accelerator == 'openvino':
            self.accelerated_model = q_model
            self.accelerate_method = 'openvino_int8'
        if accelerator is None:
            self.accelerated_model = q_model
            self.accelerate_method = 'tensorflow_int8'

    def predict(self, data, batch_size=32, quantize=False):
        if False:
            i = 10
            return i + 15
        "\n        Predict using a trained forecaster.\n\n        :params data: The data support following formats:\n\n                | 1. A numpy ndarray x:\n                | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim\n                | should be the same as past_seq_len and input_feature_num.\n                |\n                | 2. A tfdataset\n                | A TFDataset instance which contains x and y with same shape as the tuple.\n                | the tfdataset needs to return at least x in each iteration\n                | with the shape as following:\n                | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim\n                | should be the same as past_seq_len and input_feature_num.\n                | If returns x and y only get x.\n                |\n                | 3. A bigdl.chronos.data.tsdataset.TSDataset instance\n                | Forecaster will automatically process the TSDataset.\n                | By default, TSDataset will be transformed to a tfdataset,\n                | Users may call `roll` on the TSDataset before calling `fit`\n                | Then the training speed will be faster but will consume more memory.\n\n        :params batch_size: predict batch size. The value will not affect evaluate\n                result but will affect resources cost(e.g. memory and time).\n                The value default to 32. If set to None,\n                the model will be used directly for inference.\n        :param quantize: if use the quantized model to predict.\n\n        :return: A numpy array with shape (num_samples, horizon, target_dim).\n        "
        if isinstance(data, TSDataset):
            if data.lookback is None:
                data.roll(lookback=self.model_config['past_seq_len'], horizon=self.model_config['future_seq_len'])
        invalidInputError(self.fitted, 'You must call fit or restore first before calling predict!')
        if self.distributed:
            if isinstance(data, np.ndarray):
                data = np_to_xshards(data, self.workers_per_node)
            if isinstance(data, TSDataset):
                (input_data, _) = data.to_numpy()
                data = np_to_xshards(input_data, self.workers_per_node)
            invalidInputError(not isinstance(data, tf.data.Dataset), 'prediction on tf.data.Dataset will be supported in future.')
            yhat = self.internal.predict(data, batch_size=batch_size)
            yhat = xshard_to_np(yhat, mode='yhat')
        else:
            if isinstance(data, TSDataset):
                data = data.to_tf_dataset(batch_size, batch_size)
            if batch_size or isinstance(data, tf.data.Dataset):
                if quantize:
                    invalidInputError(self.accelerate_method == 'tensorflow_int8', "Can't find the quantized model, please call .quantize() method first")
                    yhat = self.accelerated_model.predict(data)
                else:
                    yhat = self.internal.predict(data)
            elif quantize:
                invalidInputError(self.accelerate_method == 'tensorflow_int8', "Can't find the quantized model, please call .quantize() method first")
                yhat = self.accelerated_model(data, training=False).numpy()
            else:
                yhat = self.internal(data, training=False).numpy()
        return yhat

    def evaluate(self, data, batch_size=32, multioutput='raw_values', quantize=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Please note that evaluate result is calculated by scaled y and yhat. If you scaled\n        your data (e.g. use .scale() on the TSDataset) please follow the following code\n        snap to evaluate your result if you need to evaluate on unscaled data.\n\n        >>> from bigdl.chronos.metric.forecaster_metrics import Evaluator\n        >>> y_hat = forecaster.predict(x)\n        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods\n        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods\n        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)\n\n        :params data: The data support following formats:\n\n                | 1. A numpy ndarray tuple (x, y):\n                | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim\n                | should be the same as past_seq_len and input_feature_num.\n                | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim\n                | should be the same as future_seq_len and output_feature_num.\n                |\n                | 2. A tf.data.Dataset:\n                | A TFDataset instance which contains x and y with same shape as the tuple.\n                | x's shape is (num_samples, lookback, feature_dim),\n                | y's shape is (num_samples, horizon, target_dim).\n                | If set distributed to True, we do not recommend using tf.data.Dataset,\n                | please replace with tsdataset or numpy.ndarray.\n                |\n                | 3. A bigdl.chronos.data.tsdataset.TSDataset instance:\n                | Forecaster will automatically process the TSDataset.\n                | By default, TSDataset will be transformed to a tfdataset,\n                | Users may call `roll` on the TSDataset before calling `fit`\n                | Then the training speed will be faster but will consume more memory.\n\n        :params batch_size: evaluate batch size. The value will not affect evaluate\n                result but will affect resources cost(e.g. memory and time).\n        :params multioutput_value: Defines aggregating of multiple output values.\n                String in ['raw_values', 'uniform_average']. The value defaults to\n                'raw_values'.The param is only effective when the forecaster is a\n                non-distribtued version.\n        :param quantize: if use the quantized model to predict.\n\n        :return: A list of evaluation results. Each item represents a metric.\n        "
        if isinstance(data, TSDataset):
            if data.lookback is None:
                data.roll(lookback=self.model_config['past_seq_len'], horizon=self.model_config['future_seq_len'])
        invalidInputError(self.fitted, 'You must call fit or restore first before calling evaluate!')
        if self.distributed:
            if isinstance(data, tuple):
                data = np_to_data_creator(data, shuffle=False)
            if isinstance(data, TSDataset):
                data = tsdata_to_data_creator(data, shuffle=False)
            invalidInputError(not isinstance(data, tf.data.Dataset), 'tf.data.Dataset is not supported, please replace with numpy.ndarray or TSDataset instance.')
            return self.internal.evaluate(data, batch_size=batch_size)
        else:
            if isinstance(data, tuple):
                (input_data, target) = data
            elif isinstance(data, TSDataset):
                (input_data, target) = data.to_numpy()
            else:
                input_data = data
                target = np.concatenate(list(map(lambda x: x[1], data.as_numpy_iterator())))
            if quantize:
                invalidInputError(self.accelerate_method == 'tensorflow_int8', "Can't find the quantized model, please call .quantize() method first")
                yhat = self.accelerated_model.predict(input_data, batch_size=batch_size)
            else:
                yhat = self.internal.predict(input_data, batch_size=batch_size)
            aggregate = 'mean' if multioutput == 'uniform_average' else None
            return Evaluator.evaluate(self.metrics, y_true=target, y_pred=yhat, aggregate=aggregate)

    def to_local(self):
        if False:
            i = 10
            return i + 15
        '\n        Transform a distributed forecaster to a local (non-distributed) one.\n\n        you need to call .to_local() and transform the forecaster to a non-\n        distributed one.\n\n        '
        if not self.distributed:
            warnings.warn('The forecaster has become local.')
        model = self.internal.get_model()
        self.internal.shutdown()
        self.internal = model
        self.distributed = False

    def get_model(self):
        if False:
            while True:
                i = 10
        '\n        Returns the learned Keras model.\n\n        :return: a keras model instance\n        '
        if self.distributed:
            return self.internal.get_model()
        else:
            return self.internal

    def save(self, checkpoint_file):
        if False:
            for i in range(10):
                print('nop')
        '\n        Save the forecaster.\n\n        :params checkpoint_file: The location you want to save the forecaster.\n        '
        invalidInputError(self.fitted, 'You must call fit or restore first before calling save!')
        self.internal.save(checkpoint_file)

    def load(self, checkpoint_file):
        if False:
            print('Hello World!')
        '\n        Load the forecaster.\n\n        :params checkpoint_file: The checkpoint file location you want to load the forecaster.\n        '
        if self.distributed:
            self.internal.load(checkpoint_file)
        else:
            self.internal = keras.models.load_model(checkpoint_file, custom_objects=self.custom_objects_config)
        self.fitted = True

    @classmethod
    def from_tsdataset(cls, tsdataset, past_seq_len=None, future_seq_len=None, **kwargs):
        if False:
            print('Hello World!')
        "\n        Build a Forecaster Model\n\n        :param tsdataset: A bigdl.chronos.data.tsdataset.TSDataset instance.\n        :param past_seq_len:  Specify history time step (i.e. lookback)\n               Do not specify the 'past_seq_len' if your tsdataset has called\n               the 'TSDataset.roll' method or 'TSDataset.to_tf_dataset'.\n        :param future_seq_len: Specify output time step (i.e. horizon)\n               Do not specify the 'future_seq_len' if your tsdataset has called\n               the 'TSDataset.roll' method or 'TSDataset.to_tf_dataset'.\n        :param kwargs: Specify parameters of Forecaster,\n               e.g. loss and optimizer, etc.\n               More info, please refer to Forecaster.__init__ methods.\n\n        :return: A Forecaster Model\n        "

        def check_time_steps(tsdataset, past_seq_len, future_seq_len):
            if False:
                for i in range(10):
                    print('nop')
            if tsdataset.lookback and past_seq_len:
                future_seq_len = future_seq_len if isinstance(future_seq_len, int) else max(future_seq_len)
                return tsdataset.lookback == past_seq_len and tsdataset.horizon == future_seq_len
            return True
        invalidInputError(not tsdataset._has_generate_agg_feature, "We will add support for 'gen_rolling_feature' method later.")
        if tsdataset.lookback:
            past_seq_len = tsdataset.lookback
            future_seq_len = tsdataset.horizon if isinstance(tsdataset.horizon, int) else max(tsdataset.horizon)
            output_feature_num = len(tsdataset.roll_target)
            input_feature_num = len(tsdataset.roll_feature) + output_feature_num
        elif past_seq_len and future_seq_len:
            past_seq_len = past_seq_len if isinstance(past_seq_len, int) else tsdataset.get_cycle_length()
            future_seq_len = future_seq_len if isinstance(future_seq_len, int) else max(future_seq_len)
            output_feature_num = len(tsdataset.target_col)
            input_feature_num = len(tsdataset.feature_col) + output_feature_num
        else:
            invalidInputError(False, "Forecaster needs 'past_seq_len' and 'future_seq_len' to specify the history time step of training.")
        invalidInputError(check_time_steps(tsdataset, past_seq_len, future_seq_len), f'tsdataset already has history time steps and differs from the given past_seq_len and future_seq_len Expected past_seq_len and future_seq_len to be {(tsdataset.lookback, tsdataset.horizon)}, but found {(past_seq_len, future_seq_len)}.', fixMsg='Do not specify past_seq_len and future seq_len or call tsdataset.roll method again and specify time step')
        return cls(past_seq_len=past_seq_len, future_seq_len=future_seq_len, input_feature_num=input_feature_num, output_feature_num=output_feature_num, **kwargs)

def _str2metric(metric):
    if False:
        print('Hello World!')
    if isinstance(metric, str):
        metric_name = metric
        from bigdl.chronos.metric.forecast_metrics import REGRESSION_MAP
        metric_func = REGRESSION_MAP[metric_name]

        def metric(y_label, y_predict):
            if False:
                print('Hello World!')
            y_label = y_label.numpy()
            y_predict = y_predict.numpy()
            return metric_func(y_label, y_predict)
        metric.__name__ = metric_name
    return metric