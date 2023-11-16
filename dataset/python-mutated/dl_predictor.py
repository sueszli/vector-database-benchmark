import abc
from typing import Dict, Optional, TypeVar, Union
import numpy as np
import pandas as pd
from ray.air.util.data_batch_conversion import BatchFormat, _convert_batch_type_to_pandas, _convert_pandas_to_batch_type
from ray.train.predictor import Predictor
from ray.util.annotations import DeveloperAPI
TensorType = TypeVar('TensorType')
TensorDtype = TypeVar('TensorDtype')

class DLPredictor(Predictor):

    @abc.abstractmethod
    def _arrays_to_tensors(self, numpy_arrays: Union[np.ndarray, Dict[str, np.ndarray]], dtype: Optional[Union[TensorDtype, Dict[str, TensorDtype]]]) -> Union[TensorType, Dict[str, TensorType]]:
        if False:
            i = 10
            return i + 15
        'Converts a NumPy ndarray batch to the tensor type for the DL framework.\n\n        Args:\n            numpy_array: The numpy array to convert to a tensor.\n            dtype: The tensor dtype to use when creating the DL tensor.\n            ndarray: A (dict of) NumPy ndarray(s) that we wish to convert to a (dict of)\n                tensor(s).\n            dtype: A (dict of) tensor dtype(s) to use when creating the DL tensor; if\n                None, the dtype will be inferred from the NumPy ndarray data.\n\n        Returns:\n            A deep learning framework specific tensor.\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def _tensor_to_array(self, tensor: TensorType) -> np.ndarray:
        if False:
            while True:
                i = 10
        'Converts tensor framework specific tensor to a numpy array.\n\n        Args:\n            tensor: A framework specific tensor.\n\n        Returns:\n            A numpy array representing the input tensor.\n        '
        raise NotImplementedError

    @abc.abstractmethod
    @DeveloperAPI
    def call_model(self, inputs: Union[TensorType, Dict[str, TensorType]]) -> Union[TensorType, Dict[str, TensorType]]:
        if False:
            i = 10
            return i + 15
        'Inputs the tensor to the model for this Predictor and returns the result.\n\n        Args:\n            inputs: The tensor to input to the model.\n\n        Returns:\n            A tensor or dictionary of tensors containing the model output.\n        '
        raise NotImplementedError

    @classmethod
    @DeveloperAPI
    def preferred_batch_format(cls) -> BatchFormat:
        if False:
            for i in range(10):
                print('nop')
        return BatchFormat.NUMPY

    def _predict_pandas(self, data: pd.DataFrame, dtype: Optional[Union[TensorDtype, Dict[str, TensorDtype]]]) -> pd.DataFrame:
        if False:
            return 10
        numpy_input = _convert_pandas_to_batch_type(data, BatchFormat.NUMPY, self._cast_tensor_columns)
        numpy_output = self._predict_numpy(numpy_input, dtype)
        return _convert_batch_type_to_pandas(numpy_output)

    def _predict_numpy(self, data: Union[np.ndarray, Dict[str, np.ndarray]], dtype: Optional[Union[TensorDtype, Dict[str, TensorDtype]]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if False:
            return 10
        if isinstance(data, dict) and len(data) == 1:
            data = next(iter(data.values()))
        model_input = self._arrays_to_tensors(data, dtype)
        model_output = self.call_model(model_input)
        if isinstance(model_output, dict):
            return {k: self._tensor_to_array(v) for (k, v) in model_output.items()}
        else:
            return {'predictions': self._tensor_to_array(model_output)}