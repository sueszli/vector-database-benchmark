from bigdl.orca.automl.model.abstract import BaseModel, ModelBuilder
import numpy as np
from bigdl.orca.automl.metrics import Evaluator
from bigdl.orca.common import SafePickle
import copy
import tensorflow as tf
from tensorflow.keras import backend as K
import types
from bigdl.dllib.utils.log4Error import *

def check_tf_version():
    if False:
        return 10
    tf_version = tf.__version__
    if tf_version >= '2.0':
        invalidInputError(False, f'Currently running TensorFlow version {tf_version}. We only supportTensorFlow 1.x for now and has been tested on 1.15')

class KerasBaseModel(BaseModel):

    def __init__(self, model_creator, check_optional_config=False):
        if False:
            print('Hello World!')
        self.check_optional_config = check_optional_config
        self.model_creator = model_creator
        self.model = None
        self.config = None
        self.model_built = False

    def build(self, config):
        if False:
            print('Hello World!')
        self._check_config(**config)
        self.config = config
        if 'selected_features' in config:
            config['input_feature_num'] = len(config['selected_features']) + config['output_feature_num']
        self.model = self.model_creator(config)
        if self.model.optimizer is None:
            invalidInputError(False, 'You must create a compiled model in model_creator')
        self.model_built = True

    @staticmethod
    def _np_to_dataset(data, batch_size):
        if False:
            i = 10
            return i + 15
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.batch(batch_size)
        return dataset

    def fit_eval(self, data, validation_data=None, mc=False, verbose=0, epochs=1, metric=None, metric_func=None, resources_per_trial=None, **config):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param data: could be a tuple with numpy ndarray with form (x, y) or\n               a data creator takes a config dict as parameter and returns a tf.data.Dataset.\n        :param validation_data: could be a tuple with numpy ndarray with form (x, y)\n        fit_eval will build a model at the first time it is built\n        config will be updated for the second or later times with only non-model-arch\n        params be functional\n        TODO: check the updated params and decide if the model is needed to be rebuilt\n        '

        def update_config():
            if False:
                i = 10
                return i + 15
            if isinstance(data, tuple) and isinstance(data[0], np.ndarray):
                (x, y) = data
                config.setdefault('input_dim', x.shape[-1])
                config.setdefault('output_dim', y.shape[-1])
                if metric and (not metric_func):
                    config.update({'metric': metric})
        if not self.model_built:
            update_config()
            self.build(config)
        else:
            tmp_config = copy.copy(self.config)
            tmp_config.update(config)
            self._check_config(**tmp_config)
            self.config.update(config)
        if isinstance(data, types.FunctionType):
            train_dataset = data(self.config)
            if validation_data:
                validation_dataset = validation_data(self.config)
            else:
                validation_dataset = validation_data
        else:
            if not isinstance(data, tuple):
                invalidInputError(False, f'data/validation_data should be a tuple of numpy array or a data creator function but found {type(data)}')
            if validation_data:
                invalidInputError(isinstance(validation_data, tuple), f'validation_data should be a tuple or data creator function but found {type(validation_data)}')
            batch_size = int(self.config.get('batch_size', 32))
            train_dataset = KerasBaseModel._np_to_dataset(data, batch_size=batch_size)
            if validation_data:
                validation_dataset = KerasBaseModel._np_to_dataset(validation_data, batch_size)
            else:
                validation_dataset = validation_data
        hist = self.model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, verbose=verbose)
        compiled_metric_names = self.model.metrics_names.copy()
        compiled_metric_names.remove('loss')
        if not metric_func:
            if not metric:
                if len(compiled_metric_names) == 1:
                    metric = compiled_metric_names[0]
                    metric_name = metric
                else:
                    invalidInputError(False, f'Got multiple metrics in compile: {compiled_metric_names}. Please choose one target metric for automl optimization')
            elif metric in compiled_metric_names:
                metric_name = metric
            else:
                try:
                    hist_metric_name = tf.keras.metrics.get(metric).__name__
                except:
                    invalidInputError(False, f'get invalid metric name {metric} for tf.keras')
                if hist_metric_name in compiled_metric_names:
                    metric_name = hist_metric_name
                else:
                    invalidInputError(False, f'Input metric in fit_eval should be one of the metrics that are used to compile the model. Got metric value of {metric} and the metrics in compile are {compiled_metric_names}')
            if validation_data is None:
                result = hist.history.get(metric_name)[-1]
            else:
                result = hist.history.get('val_' + metric_name)[-1]
            return {metric: result}
        else:
            metric_name = metric or metric_func.__name__
            if validation_data is not None:
                val_x = validation_data[0]
                val_y = validation_data[1]
            else:
                val_x = data[0]
                val_y = data[1]
            y_pred = self.predict(val_x)
            result = metric_func(val_y, y_pred)
            return {metric_name: result}

    def evaluate(self, x, y, batch_size=32, metrics=['mse'], multioutput='raw_values'):
        if False:
            while True:
                i = 10
        '\n        Evaluate on x, y\n        :param x: input\n        :param y: target\n        :param metrics: a list of metrics in string format\n        :param multioutput: output mode\n        :return: a list of metric evaluation results\n        '
        y_pred = self.predict(x, batch_size=batch_size)
        return [Evaluator.evaluate(m, y, y_pred, multioutput=multioutput) for m in metrics]

    def predict(self, x, batch_size=32):
        if False:
            while True:
                i = 10
        '\n        Prediction on x.\n        :param x: input\n        :param batch_size: batch\n        :return: predicted y\n        '
        if not self.model_built:
            invalidInputError(False, 'You must call fit_eval or restore first before calling predict!')
        return self.model.predict(x, batch_size=batch_size)

    def predict_with_uncertainty(self, x, n_iter=100):
        if False:
            i = 10
            return i + 15
        if not self.model_built:
            invalidInputError(False, 'You must call fit_eval or restore first before calling predict!')
        check_tf_version()
        f = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[-1].output])
        result = np.array([f((x, 1))[0] for _ in range(n_iter)])
        prediction = result.mean(axis=0)
        uncertainty = result.var(axis=0)
        return (prediction, uncertainty)

    def state_dict(self):
        if False:
            while True:
                i = 10
        state = {'config': self.config, 'weights': self.model.get_weights()}
        return state

    def load_state_dict(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.config = state['config']
        self.model = self.model_creator(self.config)
        self.model.set_weights(state['weights'])
        self.model_built = True

    def save(self, checkpoint):
        if False:
            return 10
        if not self.model_built:
            invalidInputError(False, 'You must call fit_eval or restore first before calling save!')
        state_dict = self.state_dict()
        with open(checkpoint, 'wb') as f:
            SafePickle.dump(state_dict, f)

    def restore(self, checkpoint):
        if False:
            while True:
                i = 10
        with open(checkpoint, 'rb') as f:
            state_dict = SafePickle.load(f)
        self.load_state_dict(state_dict)

    def _get_required_parameters(self):
        if False:
            print('Hello World!')
        return set()

    def _get_optional_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        return {'batch_size'}

class KerasModelBuilder(ModelBuilder):

    def __init__(self, model_creator):
        if False:
            for i in range(10):
                print('nop')
        self.model_creator = model_creator

    def build(self, config):
        if False:
            for i in range(10):
                print('nop')
        model = KerasBaseModel(self.model_creator)
        model.build(config)
        return model