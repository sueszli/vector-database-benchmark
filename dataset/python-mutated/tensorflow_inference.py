import enum
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Union
import numpy
import tensorflow as tf
import tensorflow_hub as hub
from apache_beam.ml.inference import utils
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import PredictionResult
__all__ = ['TFModelHandlerNumpy', 'TFModelHandlerTensor']
TensorInferenceFn = Callable[[tf.Module, Sequence[Union[numpy.ndarray, tf.Tensor]], Dict[str, Any], Optional[str]], Iterable[PredictionResult]]

class ModelType(enum.Enum):
    """Defines how a model file should be loaded."""
    SAVED_MODEL = 1
    SAVED_WEIGHTS = 2

def _load_model(model_uri, custom_weights, load_model_args):
    if False:
        return 10
    try:
        model = tf.keras.models.load_model(hub.resolve(model_uri), **load_model_args)
    except Exception as e:
        raise ValueError("Unable to load the TensorFlow model: {exception}. Make sure you've         saved the model with TF2 format. Check out the list of TF2 Models on         TensorFlow Hub - https://tfhub.dev/s?subtype=module,placeholder&tf-version=tf2.".format(exception=e))
    if custom_weights:
        model.load_weights(custom_weights)
    return model

def _load_model_from_weights(create_model_fn, weights_path):
    if False:
        i = 10
        return i + 15
    model = create_model_fn()
    model.load_weights(weights_path)
    return model

def default_numpy_inference_fn(model: tf.Module, batch: Sequence[numpy.ndarray], inference_args: Dict[str, Any], model_id: Optional[str]=None) -> Iterable[PredictionResult]:
    if False:
        return 10
    vectorized_batch = numpy.stack(batch, axis=0)
    predictions = model(vectorized_batch, **inference_args)
    return utils._convert_to_result(batch, predictions, model_id)

def default_tensor_inference_fn(model: tf.Module, batch: Sequence[tf.Tensor], inference_args: Dict[str, Any], model_id: Optional[str]=None) -> Iterable[PredictionResult]:
    if False:
        while True:
            i = 10
    vectorized_batch = tf.stack(batch, axis=0)
    predictions = model(vectorized_batch, **inference_args)
    return utils._convert_to_result(batch, predictions, model_id)

class TFModelHandlerNumpy(ModelHandler[numpy.ndarray, PredictionResult, tf.Module]):

    def __init__(self, model_uri: str, model_type: ModelType=ModelType.SAVED_MODEL, create_model_fn: Optional[Callable]=None, *, load_model_args: Optional[Dict[str, Any]]=None, custom_weights: str='', inference_fn: TensorInferenceFn=default_numpy_inference_fn, min_batch_size: Optional[int]=None, max_batch_size: Optional[int]=None, large_model: bool=False, **kwargs):
        if False:
            return 10
        'Implementation of the ModelHandler interface for Tensorflow.\n\n    Example Usage::\n\n      pcoll | RunInference(TFModelHandlerNumpy(model_uri="my_uri"))\n\n    See https://www.tensorflow.org/tutorials/keras/save_and_load for details.\n\n    Args:\n        model_uri (str): path to the trained model.\n        model_type: type of model to be loaded. Defaults to SAVED_MODEL.\n        create_model_fn: a function that creates and returns a new\n          tensorflow model to load the saved weights.\n          It should be used with ModelType.SAVED_WEIGHTS.\n        load_model_args: a dictionary of parameters to pass to the load_model\n          function of TensorFlow to specify custom config.\n        custom_weights (str): path to the custom weights to be applied\n          once the model is loaded.\n        inference_fn: inference function to use during RunInference.\n          Defaults to default_numpy_inference_fn.\n        large_model: set to true if your model is large enough to run into\n          memory pressure if you load multiple copies. Given a model that\n          consumes N memory and a machine with W cores and M memory, you should\n          set this to True if N*W > M.\n        kwargs: \'env_vars\' can be used to set environment variables\n          before loading the model.\n\n    **Supported Versions:** RunInference APIs in Apache Beam have been tested\n    with Tensorflow 2.9, 2.10, 2.11.\n    '
        self._model_uri = model_uri
        self._model_type = model_type
        self._inference_fn = inference_fn
        self._create_model_fn = create_model_fn
        self._env_vars = kwargs.get('env_vars', {})
        self._load_model_args = {} if not load_model_args else load_model_args
        self._custom_weights = custom_weights
        self._batching_kwargs = {}
        if min_batch_size is not None:
            self._batching_kwargs['min_batch_size'] = min_batch_size
        if max_batch_size is not None:
            self._batching_kwargs['max_batch_size'] = max_batch_size
        self._large_model = large_model

    def load_model(self) -> tf.Module:
        if False:
            i = 10
            return i + 15
        'Loads and initializes a Tensorflow model for processing.'
        if self._model_type == ModelType.SAVED_WEIGHTS:
            if not self._create_model_fn:
                raise ValueError('Callable create_model_fn must be passedwith ModelType.SAVED_WEIGHTS')
            return _load_model_from_weights(self._create_model_fn, self._model_uri)
        return _load_model(self._model_uri, self._custom_weights, self._load_model_args)

    def update_model_path(self, model_path: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        self._model_uri = model_path if model_path else self._model_uri

    def run_inference(self, batch: Sequence[numpy.ndarray], model: tf.Module, inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            for i in range(10):
                print('nop')
        "\n    Runs inferences on a batch of numpy array and returns an Iterable of\n    numpy array Predictions.\n\n    This method stacks the n-dimensional numpy array in a vectorized format to\n    optimize the inference call.\n\n    Args:\n      batch: A sequence of numpy nd-array. These should be batchable, as this\n        method will call `numpy.stack()` and pass in batched numpy nd-array\n        with dimensions (batch_size, n_features, etc.) into the model's\n        predict() function.\n      model: A Tensorflow model.\n      inference_args: any additional arguments for an inference.\n\n    Returns:\n      An Iterable of type PredictionResult.\n    "
        inference_args = {} if not inference_args else inference_args
        return self._inference_fn(model, batch, inference_args, self._model_uri)

    def get_num_bytes(self, batch: Sequence[numpy.ndarray]) -> int:
        if False:
            return 10
        '\n    Returns:\n      The number of bytes of data for a batch of numpy arrays.\n    '
        return sum((sys.getsizeof(element) for element in batch))

    def get_metrics_namespace(self) -> str:
        if False:
            print('Hello World!')
        '\n    Returns:\n       A namespace for metrics collected by the RunInference transform.\n    '
        return 'BeamML_TF_Numpy'

    def validate_inference_args(self, inference_args: Optional[Dict[str, Any]]):
        if False:
            print('Hello World!')
        pass

    def batch_elements_kwargs(self):
        if False:
            while True:
                i = 10
        return self._batching_kwargs

    def share_model_across_processes(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._large_model

class TFModelHandlerTensor(ModelHandler[tf.Tensor, PredictionResult, tf.Module]):

    def __init__(self, model_uri: str, model_type: ModelType=ModelType.SAVED_MODEL, create_model_fn: Optional[Callable]=None, *, load_model_args: Optional[Dict[str, Any]]=None, custom_weights: str='', inference_fn: TensorInferenceFn=default_tensor_inference_fn, min_batch_size: Optional[int]=None, max_batch_size: Optional[int]=None, large_model: bool=False, **kwargs):
        if False:
            print('Hello World!')
        'Implementation of the ModelHandler interface for Tensorflow.\n\n    Example Usage::\n\n      pcoll | RunInference(TFModelHandlerTensor(model_uri="my_uri"))\n\n    See https://www.tensorflow.org/tutorials/keras/save_and_load for details.\n\n    Args:\n        model_uri (str): path to the trained model.\n        model_type: type of model to be loaded.\n          Defaults to SAVED_MODEL.\n        create_model_fn: a function that creates and returns a new\n          tensorflow model to load the saved weights.\n          It should be used with ModelType.SAVED_WEIGHTS.\n        load_model_args: a dictionary of parameters to pass to the load_model\n          function of TensorFlow to specify custom config.\n        custom_weights (str): path to the custom weights to be applied\n          once the model is loaded.\n        inference_fn: inference function to use during RunInference.\n          Defaults to default_numpy_inference_fn.\n        large_model: set to true if your model is large enough to run into\n          memory pressure if you load multiple copies. Given a model that\n          consumes N memory and a machine with W cores and M memory, you should\n          set this to True if N*W > M.\n        kwargs: \'env_vars\' can be used to set environment variables\n          before loading the model.\n\n    **Supported Versions:** RunInference APIs in Apache Beam have been tested\n    with Tensorflow 2.11.\n    '
        self._model_uri = model_uri
        self._model_type = model_type
        self._inference_fn = inference_fn
        self._create_model_fn = create_model_fn
        self._env_vars = kwargs.get('env_vars', {})
        self._load_model_args = {} if not load_model_args else load_model_args
        self._custom_weights = custom_weights
        self._batching_kwargs = {}
        if min_batch_size is not None:
            self._batching_kwargs['min_batch_size'] = min_batch_size
        if max_batch_size is not None:
            self._batching_kwargs['max_batch_size'] = max_batch_size
        self._large_model = large_model

    def load_model(self) -> tf.Module:
        if False:
            i = 10
            return i + 15
        'Loads and initializes a tensorflow model for processing.'
        if self._model_type == ModelType.SAVED_WEIGHTS:
            if not self._create_model_fn:
                raise ValueError('Callable create_model_fn must be passedwith ModelType.SAVED_WEIGHTS')
            return _load_model_from_weights(self._create_model_fn, self._model_uri)
        return _load_model(self._model_uri, self._custom_weights, self._load_model_args)

    def update_model_path(self, model_path: Optional[str]=None):
        if False:
            return 10
        self._model_uri = model_path if model_path else self._model_uri

    def run_inference(self, batch: Sequence[tf.Tensor], model: tf.Module, inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            for i in range(10):
                print('nop')
        "\n    Runs inferences on a batch of tf.Tensor and returns an Iterable of\n    Tensor Predictions.\n\n    This method stacks the list of Tensors in a vectorized format to optimize\n    the inference call.\n\n    Args:\n      batch: A sequence of Tensors. These Tensors should be batchable, as this\n        method will call `tf.stack()` and pass in batched Tensors with\n        dimensions (batch_size, n_features, etc.) into the model's predict()\n        function.\n      model: A Tensorflow model.\n      inference_args: Non-batchable arguments required as inputs to the model's\n        forward() function. Unlike Tensors in `batch`, these parameters will\n        not be dynamically batched\n    Returns:\n      An Iterable of type PredictionResult.\n    "
        inference_args = {} if not inference_args else inference_args
        return self._inference_fn(model, batch, inference_args, self._model_uri)

    def get_num_bytes(self, batch: Sequence[tf.Tensor]) -> int:
        if False:
            return 10
        '\n    Returns:\n      The number of bytes of data for a batch of Tensors.\n    '
        return sum((sys.getsizeof(element) for element in batch))

    def get_metrics_namespace(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n    Returns:\n       A namespace for metrics collected by the RunInference transform.\n    '
        return 'BeamML_TF_Tensor'

    def validate_inference_args(self, inference_args: Optional[Dict[str, Any]]):
        if False:
            while True:
                i = 10
        pass

    def batch_elements_kwargs(self):
        if False:
            i = 10
            return i + 15
        return self._batching_kwargs

    def share_model_across_processes(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._large_model