import enum
import pickle
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
import numpy
import pandas
from sklearn.base import BaseEstimator
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference import utils
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import PredictionResult
try:
    import joblib
except ImportError:
    pass
__all__ = ['SklearnModelHandlerNumpy', 'SklearnModelHandlerPandas']
NumpyInferenceFn = Callable[[BaseEstimator, Sequence[numpy.ndarray], Optional[Dict[str, Any]]], Any]

class ModelFileType(enum.Enum):
    """Defines how a model file is serialized. Options are pickle or joblib."""
    PICKLE = 1
    JOBLIB = 2

def _load_model(model_uri, file_type):
    if False:
        for i in range(10):
            print('nop')
    file = FileSystems.open(model_uri, 'rb')
    if file_type == ModelFileType.PICKLE:
        return pickle.load(file)
    elif file_type == ModelFileType.JOBLIB:
        if not joblib:
            raise ImportError('Could not import joblib in this execution environment. For help with managing dependencies on Python workers.see https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/')
        return joblib.load(file)
    raise AssertionError('Unsupported serialization type.')

def _default_numpy_inference_fn(model: BaseEstimator, batch: Sequence[numpy.ndarray], inference_args: Optional[Dict[str, Any]]=None) -> Any:
    if False:
        i = 10
        return i + 15
    vectorized_batch = numpy.stack(batch, axis=0)
    return model.predict(vectorized_batch)

class SklearnModelHandlerNumpy(ModelHandler[numpy.ndarray, PredictionResult, BaseEstimator]):

    def __init__(self, model_uri: str, model_file_type: ModelFileType=ModelFileType.PICKLE, *, inference_fn: NumpyInferenceFn=_default_numpy_inference_fn, min_batch_size: Optional[int]=None, max_batch_size: Optional[int]=None, large_model: bool=False, **kwargs):
        if False:
            print('Hello World!')
        ' Implementation of the ModelHandler interface for scikit-learn\n    using numpy arrays as input.\n\n    Example Usage::\n\n      pcoll | RunInference(SklearnModelHandlerNumpy(model_uri="my_uri"))\n\n    Args:\n      model_uri: The URI to where the model is saved.\n      model_file_type: The method of serialization of the argument.\n        default=pickle\n      inference_fn: The inference function to use.\n        default=_default_numpy_inference_fn\n      min_batch_size: the minimum batch size to use when batching inputs. This\n        batch will be fed into the inference_fn as a Sequence of Numpy\n        ndarrays.\n      max_batch_size: the maximum batch size to use when batching inputs. This\n        batch will be fed into the inference_fn as a Sequence of Numpy\n        ndarrays.\n      large_model: set to true if your model is large enough to run into\n        memory pressure if you load multiple copies. Given a model that\n        consumes N memory and a machine with W cores and M memory, you should\n        set this to True if N*W > M.\n      kwargs: \'env_vars\' can be used to set environment variables\n        before loading the model.\n    '
        self._model_uri = model_uri
        self._model_file_type = model_file_type
        self._model_inference_fn = inference_fn
        self._batching_kwargs = {}
        if min_batch_size is not None:
            self._batching_kwargs['min_batch_size'] = min_batch_size
        if max_batch_size is not None:
            self._batching_kwargs['max_batch_size'] = max_batch_size
        self._env_vars = kwargs.get('env_vars', {})
        self._large_model = large_model

    def load_model(self) -> BaseEstimator:
        if False:
            return 10
        'Loads and initializes a model for processing.'
        return _load_model(self._model_uri, self._model_file_type)

    def update_model_path(self, model_path: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        self._model_uri = model_path if model_path else self._model_uri

    def run_inference(self, batch: Sequence[numpy.ndarray], model: BaseEstimator, inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            i = 10
            return i + 15
        'Runs inferences on a batch of numpy arrays.\n\n    Args:\n      batch: A sequence of examples as numpy arrays. They should\n        be single examples.\n      model: A numpy model or pipeline. Must implement predict(X).\n        Where the parameter X is a numpy array.\n      inference_args: Any additional arguments for an inference.\n\n    Returns:\n      An Iterable of type PredictionResult.\n    '
        predictions = self._model_inference_fn(model, batch, inference_args)
        return utils._convert_to_result(batch, predictions, model_id=self._model_uri)

    def get_num_bytes(self, batch: Sequence[numpy.ndarray]) -> int:
        if False:
            while True:
                i = 10
        '\n    Returns:\n      The number of bytes of data for a batch.\n    '
        return sum((sys.getsizeof(element) for element in batch))

    def get_metrics_namespace(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n    Returns:\n       A namespace for metrics collected by the RunInference transform.\n    '
        return 'BeamML_Sklearn'

    def batch_elements_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        return self._batching_kwargs

    def share_model_across_processes(self) -> bool:
        if False:
            while True:
                i = 10
        return self._large_model
PandasInferenceFn = Callable[[BaseEstimator, Sequence[pandas.DataFrame], Optional[Dict[str, Any]]], Any]

def _default_pandas_inference_fn(model: BaseEstimator, batch: Sequence[pandas.DataFrame], inference_args: Optional[Dict[str, Any]]=None) -> Any:
    if False:
        while True:
            i = 10
    vectorized_batch = pandas.concat(batch, axis=0)
    predictions = model.predict(vectorized_batch)
    splits = [vectorized_batch.iloc[[i]] for i in range(vectorized_batch.shape[0])]
    return (predictions, splits)

class SklearnModelHandlerPandas(ModelHandler[pandas.DataFrame, PredictionResult, BaseEstimator]):

    def __init__(self, model_uri: str, model_file_type: ModelFileType=ModelFileType.PICKLE, *, inference_fn: PandasInferenceFn=_default_pandas_inference_fn, min_batch_size: Optional[int]=None, max_batch_size: Optional[int]=None, large_model: bool=False, **kwargs):
        if False:
            print('Hello World!')
        'Implementation of the ModelHandler interface for scikit-learn that\n    supports pandas dataframes.\n\n    Example Usage::\n\n      pcoll | RunInference(SklearnModelHandlerPandas(model_uri="my_uri"))\n\n    **NOTE:** This API and its implementation are under development and\n    do not provide backward compatibility guarantees.\n\n    Args:\n      model_uri: The URI to where the model is saved.\n      model_file_type: The method of serialization of the argument.\n        default=pickle\n      inference_fn: The inference function to use.\n        default=_default_pandas_inference_fn\n      min_batch_size: the minimum batch size to use when batching inputs. This\n        batch will be fed into the inference_fn as a Sequence of Pandas\n        Dataframes.\n      max_batch_size: the maximum batch size to use when batching inputs. This\n        batch will be fed into the inference_fn as a Sequence of Pandas\n        Dataframes.\n      large_model: set to true if your model is large enough to run into\n        memory pressure if you load multiple copies. Given a model that\n        consumes N memory and a machine with W cores and M memory, you should\n        set this to True if N*W > M.\n      kwargs: \'env_vars\' can be used to set environment variables\n        before loading the model.\n    '
        self._model_uri = model_uri
        self._model_file_type = model_file_type
        self._model_inference_fn = inference_fn
        self._batching_kwargs = {}
        if min_batch_size is not None:
            self._batching_kwargs['min_batch_size'] = min_batch_size
        if max_batch_size is not None:
            self._batching_kwargs['max_batch_size'] = max_batch_size
        self._env_vars = kwargs.get('env_vars', {})
        self._large_model = large_model

    def load_model(self) -> BaseEstimator:
        if False:
            print('Hello World!')
        'Loads and initializes a model for processing.'
        return _load_model(self._model_uri, self._model_file_type)

    def update_model_path(self, model_path: Optional[str]=None):
        if False:
            print('Hello World!')
        self._model_uri = model_path if model_path else self._model_uri

    def run_inference(self, batch: Sequence[pandas.DataFrame], model: BaseEstimator, inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            return 10
        '\n    Runs inferences on a batch of pandas dataframes.\n\n    Args:\n      batch: A sequence of examples as numpy arrays. They should\n        be single examples.\n      model: A dataframe model or pipeline. Must implement predict(X).\n        Where the parameter X is a pandas dataframe.\n      inference_args: Any additional arguments for an inference.\n\n    Returns:\n      An Iterable of type PredictionResult.\n    '
        for dataframe in iter(batch):
            if dataframe.shape[0] != 1:
                raise ValueError('Only dataframes with single rows are supported.')
        (predictions, splits) = self._model_inference_fn(model, batch, inference_args)
        return utils._convert_to_result(splits, predictions, model_id=self._model_uri)

    def get_num_bytes(self, batch: Sequence[pandas.DataFrame]) -> int:
        if False:
            while True:
                i = 10
        '\n    Returns:\n      The number of bytes of data for a batch.\n    '
        return sum((df.memory_usage(deep=True).sum() for df in batch))

    def get_metrics_namespace(self) -> str:
        if False:
            print('Hello World!')
        '\n    Returns:\n       A namespace for metrics collected by the RunInference transform.\n    '
        return 'BeamML_Sklearn'

    def batch_elements_kwargs(self):
        if False:
            print('Hello World!')
        return self._batching_kwargs

    def share_model_across_processes(self) -> bool:
        if False:
            return 10
        return self._large_model