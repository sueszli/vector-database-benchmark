"""
This module implements the abstract base class for the ablators.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    import tensorflow as tf
    import torch

class BaseAblator(ABC):
    """
    Base class defining the methods used for the ablators.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray, column_pos: Optional[Union[int, list]]=None, row_pos: Optional[Union[int, list]]=None) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Ablate the image x at location specified by "column_pos" for the case of column ablation or at the location\n        specified by "column_pos" and "row_pos" in the case of block ablation.\n\n        :param x: input image.\n        :param column_pos: column position to specify where to retain the image\n        :param row_pos: row position to specify where to retain the image. Not used for ablation type "column".\n        '
        raise NotImplementedError

    @abstractmethod
    def certify(self, pred_counts: np.ndarray, size_to_certify: int, label: Union[np.ndarray, 'tf.Tensor']) -> Union[Tuple['tf.Tensor', 'tf.Tensor', 'tf.Tensor'], Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor']]:
        if False:
            while True:
                i = 10
        '\n        Checks if based on the predictions supplied the classifications over the ablated datapoints result in a\n        certified prediction against a patch attack of size size_to_certify.\n\n        :param pred_counts: The cumulative predictions of the classifier over the ablation locations.\n        :param size_to_certify: The size of the patch to check against.\n        :param label: ground truth labels\n        '
        raise NotImplementedError

    @abstractmethod
    def ablate(self, x: np.ndarray, column_pos: int, row_pos: int) -> Union[np.ndarray, 'torch.Tensor']:
        if False:
            for i in range(10):
                print('nop')
        '\n        Ablate the image x at location specified by "column_pos" for the case of column ablation or at the location\n        specified by "column_pos" and "row_pos" in the case of block ablation.\n\n        :param x: input image.\n        :param column_pos: column position to specify where to retain the image\n        :param row_pos: row position to specify where to retain the image. Not used for ablation type "column".\n        '
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: np.ndarray, column_pos: Optional[int]=None, row_pos: Optional[int]=None) -> Union[np.ndarray, 'torch.Tensor']:
        if False:
            i = 10
            return i + 15
        '\n        Ablate batch of data at locations specified by column_pos and row_pos\n\n        :param x: input image.\n        :param column_pos: column position to specify where to retain the image\n        :param row_pos: row position to specify where to retain the image. Not used for ablation type "column".\n        '
        raise NotImplementedError