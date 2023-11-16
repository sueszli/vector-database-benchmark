from abc import ABC, abstractmethod
import pandas as pd
import torch

class PyTorchDataConvertor(ABC):
    """
    This class is responsible for converting `*_features` & `*_labels` pandas dataframes
    to pytorch tensors.
    """

    @abstractmethod
    def convert_x(self, df: pd.DataFrame, device: str) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        :param df: "*_features" dataframe.\n        :param device: The device to use for training (e.g. \'cpu\', \'cuda\').\n        '

    @abstractmethod
    def convert_y(self, df: pd.DataFrame, device: str) -> torch.Tensor:
        if False:
            return 10
        '\n        :param df: "*_labels" dataframe.\n        :param device: The device to use for training (e.g. \'cpu\', \'cuda\').\n        '

class DefaultPyTorchDataConvertor(PyTorchDataConvertor):
    """
    A default conversion that keeps features dataframe shapes.
    """

    def __init__(self, target_tensor_type: torch.dtype=torch.float32, squeeze_target_tensor: bool=False):
        if False:
            return 10
        '\n        :param target_tensor_type: type of target tensor, for classification use\n            torch.long, for regressor use torch.float or torch.double.\n        :param squeeze_target_tensor: controls the target shape, used for loss functions\n            that requires 0D or 1D.\n        '
        self._target_tensor_type = target_tensor_type
        self._squeeze_target_tensor = squeeze_target_tensor

    def convert_x(self, df: pd.DataFrame, device: str) -> torch.Tensor:
        if False:
            return 10
        numpy_arrays = df.values
        x = torch.tensor(numpy_arrays, device=device, dtype=torch.float32)
        return x

    def convert_y(self, df: pd.DataFrame, device: str) -> torch.Tensor:
        if False:
            while True:
                i = 10
        numpy_arrays = df.values
        y = torch.tensor(numpy_arrays, device=device, dtype=self._target_tensor_type)
        if self._squeeze_target_tensor:
            y = y.squeeze()
        return y