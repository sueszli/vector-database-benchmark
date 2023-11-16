import sys
from abc import ABC
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Union
import numpy
import pandas
import scipy
import datatable
import xgboost
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import ExampleT
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import ModelT
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.base import PredictionT
__all__ = ['XGBoostModelHandler', 'XGBoostModelHandlerNumpy', 'XGBoostModelHandlerPandas', 'XGBoostModelHandlerSciPy', 'XGBoostModelHandlerDatatable']
XGBoostInferenceFn = Callable[[Sequence[object], Union[xgboost.Booster, xgboost.XGBModel], Optional[Dict[str, Any]]], Iterable[PredictionResult]]

def default_xgboost_inference_fn(batch: Sequence[object], model: Union[xgboost.Booster, xgboost.XGBModel], inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
    if False:
        for i in range(10):
            print('nop')
    inference_args = {} if not inference_args else inference_args
    if type(model) == xgboost.Booster:
        batch = [xgboost.DMatrix(array) for array in batch]
    predictions = [model.predict(el, **inference_args) for el in batch]
    return [PredictionResult(x, y) for (x, y) in zip(batch, predictions)]

class XGBoostModelHandler(ModelHandler[ExampleT, PredictionT, ModelT], ABC):

    def __init__(self, model_class: Union[Callable[..., xgboost.Booster], Callable[..., xgboost.XGBModel]], model_state: str, inference_fn: XGBoostInferenceFn=default_xgboost_inference_fn, **kwargs):
        if False:
            while True:
                i = 10
        'Implementation of the ModelHandler interface for XGBoost.\n\n    Example Usage::\n\n        pcoll | RunInference(\n                    XGBoostModelHandler(\n                        model_class="XGBoost Model Class",\n                        model_state="my_model_state.json")))\n\n    See https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html\n    for details\n\n    Args:\n      model_class: class of the XGBoost model that defines the model\n        structure.\n      model_state: path to a json file that contains the model\'s\n        configuration.\n      inference_fn: the inference function to use during RunInference.\n        default=default_xgboost_inference_fn\n      kwargs: \'env_vars\' can be used to set environment variables\n        before loading the model.\n\n    **Supported Versions:** RunInference APIs in Apache Beam have been tested\n    with XGBoost 1.6.0 and 1.7.0\n\n    XGBoost 1.0.0 introduced support for using JSON to save and load\n    XGBoost models. XGBoost 1.6.0, additional support for Universal Binary JSON.\n    It is recommended to use a model trained in XGBoost 1.6.0 or higher.\n    While you should be able to load models created in older versions, there\n    are no guarantees this will work as expected.\n\n    This class is the superclass of all the various XGBoostModelhandlers\n    and should not be instantiated directly. (See instead\n    XGBoostModelHandlerNumpy, XGBoostModelHandlerPandas, etc.)\n    '
        self._model_class = model_class
        self._model_state = model_state
        self._inference_fn = inference_fn
        self._env_vars = kwargs.get('env_vars', {})

    def load_model(self) -> Union[xgboost.Booster, xgboost.XGBModel]:
        if False:
            while True:
                i = 10
        model = self._model_class()
        model_state_file_handler = FileSystems.open(self._model_state, 'rb')
        model_state_bytes = model_state_file_handler.read()
        model_state_bytearray = bytearray(model_state_bytes)
        model.load_model(model_state_bytearray)
        return model

    def get_metrics_namespace(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'BeamML_XGBoost'

class XGBoostModelHandlerNumpy(XGBoostModelHandler[numpy.ndarray, PredictionResult, Union[xgboost.Booster, xgboost.XGBModel]]):
    """Implementation of the ModelHandler interface for XGBoost
  using numpy arrays as input.

  Example Usage::

      pcoll | RunInference(
                  XGBoostModelHandlerNumpy(
                      model_class="XGBoost Model Class",
                      model_state="my_model_state.json")))

  Args:
    model_class: class of the XGBoost model that defines the model
      structure.
    model_state: path to a json file that contains the model's
      configuration.
    inference_fn: the inference function to use during RunInference.
      default=default_xgboost_inference_fn
  """

    def run_inference(self, batch: Sequence[numpy.ndarray], model: Union[xgboost.Booster, xgboost.XGBModel], inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            for i in range(10):
                print('nop')
        'Runs inferences on a batch of 2d numpy arrays.\n\n    Args:\n      batch: A sequence of examples as 2d numpy arrays. Each\n        row in an array is a single example. The dimensions\n        must match the dimensions of the data used to train\n        the model.\n      model: XGBoost booster or XBGModel (sklearn interface). Must\n        implement predict(X). Where the parameter X is a 2d numpy array.\n      inference_args: Any additional arguments for an inference.\n\n    Returns:\n      An Iterable of type PredictionResult.\n    '
        return self._inference_fn(batch, model, inference_args)

    def get_num_bytes(self, batch: Sequence[numpy.ndarray]) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n    Returns:\n      The number of bytes of data for a batch.\n    '
        return sum((sys.getsizeof(element) for element in batch))

class XGBoostModelHandlerPandas(XGBoostModelHandler[pandas.DataFrame, PredictionResult, Union[xgboost.Booster, xgboost.XGBModel]]):
    """Implementation of the ModelHandler interface for XGBoost
  using pandas dataframes as input.

  Example Usage::

      pcoll | RunInference(
                  XGBoostModelHandlerPandas(
                      model_class="XGBoost Model Class",
                      model_state="my_model_state.json")))

  Args:
    model_class: class of the XGBoost model that defines the model
      structure.
    model_state: path to a json file that contains the model's
      configuration.
    inference_fn: the inference function to use during RunInference.
      default=default_xgboost_inference_fn
  """

    def run_inference(self, batch: Sequence[pandas.DataFrame], model: Union[xgboost.Booster, xgboost.XGBModel], inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            i = 10
            return i + 15
        'Runs inferences on a batch of pandas dataframes.\n\n    Args:\n      batch: A sequence of examples as pandas dataframes. Each\n        row in a dataframe is a single example. The dimensions\n        must match the dimensions of the data used to train\n        the model.\n      model: XGBoost booster or XBGModel (sklearn interface). Must\n        implement predict(X). Where the parameter X is a pandas dataframe.\n      inference_args: Any additional arguments for an inference.\n\n    Returns:\n      An Iterable of type PredictionResult.\n    '
        return self._inference_fn(batch, model, inference_args)

    def get_num_bytes(self, batch: Sequence[pandas.DataFrame]) -> int:
        if False:
            return 10
        '\n    Returns:\n        The number of bytes of data for a batch of Numpy arrays.\n    '
        return sum((df.memory_usage(deep=True).sum() for df in batch))

class XGBoostModelHandlerSciPy(XGBoostModelHandler[scipy.sparse.csr_matrix, PredictionResult, Union[xgboost.Booster, xgboost.XGBModel]]):
    """ Implementation of the ModelHandler interface for XGBoost
  using scipy matrices as input.

  Example Usage::

      pcoll | RunInference(
                  XGBoostModelHandlerSciPy(
                      model_class="XGBoost Model Class",
                      model_state="my_model_state.json")))

  Args:
    model_class: class of the XGBoost model that defines the model
      structure.
    model_state: path to a json file that contains the model's
      configuration.
    inference_fn: the inference function to use during RunInference.
      default=default_xgboost_inference_fn
  """

    def run_inference(self, batch: Sequence[scipy.sparse.csr_matrix], model: Union[xgboost.Booster, xgboost.XGBModel], inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            for i in range(10):
                print('nop')
        'Runs inferences on a batch of SciPy sparse matrices.\n\n    Args:\n      batch: A sequence of examples as Scipy sparse matrices.\n       The dimensions must match the dimensions of the data\n       used to train the model.\n      model: XGBoost booster or XBGModel (sklearn interface). Must implement\n        predict(X). Where the parameter X is a SciPy sparse matrix.\n      inference_args: Any additional arguments for an inference.\n\n    Returns:\n      An Iterable of type PredictionResult.\n    '
        return self._inference_fn(batch, model, inference_args)

    def get_num_bytes(self, batch: Sequence[scipy.sparse.csr_matrix]) -> int:
        if False:
            i = 10
            return i + 15
        '\n    Returns:\n      The number of bytes of data for a batch.\n    '
        return sum((sys.getsizeof(element) for element in batch))

class XGBoostModelHandlerDatatable(XGBoostModelHandler[datatable.Frame, PredictionResult, Union[xgboost.Booster, xgboost.XGBModel]]):
    """Implementation of the ModelHandler interface for XGBoost
  using datatable dataframes as input.

  Example Usage::

      pcoll | RunInference(
                  XGBoostModelHandlerDatatable(
                      model_class="XGBoost Model Class",
                      model_state="my_model_state.json")))

  Args:
    model_class: class of the XGBoost model that defines the model
      structure.
    model_state: path to a json file that contains the model's
      configuration.
    inference_fn: the inference function to use during RunInference.
      default=default_xgboost_inference_fn
  """

    def run_inference(self, batch: Sequence[datatable.Frame], model: Union[xgboost.Booster, xgboost.XGBModel], inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            return 10
        'Runs inferences on a batch of datatable dataframe.\n\n    Args:\n      batch: A sequence of examples as datatable dataframes. Each\n        row in a dataframe is a single example. The dimensions\n        must match the dimensions of the data used to train\n        the model.\n      model: XGBoost booster or XBGModel (sklearn interface). Must implement\n        predict(X). Where the parameter X is a datatable dataframe.\n      inference_args: Any additional arguments for an inference.\n\n    Returns:\n      An Iterable of type PredictionResult.\n    '
        return self._inference_fn(batch, model, inference_args)

    def get_num_bytes(self, batch: Sequence[datatable.Frame]) -> int:
        if False:
            print('Hello World!')
        '\n    Returns:\n      The number of bytes of data for a batch.\n    '
        return sum((sys.getsizeof(element) for element in batch))