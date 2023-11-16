from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from modelscope.models.base.base_model import Model
from modelscope.utils.config import ConfigDict
from modelscope.utils.logger import get_logger
logger = get_logger()
Tensor = Union['torch.Tensor', 'tf.Tensor']
Input = Union[Dict[str, Tensor], Model]

class Head(ABC):
    """The head base class is for the tasks head method definition

    """

    def __init__(self, **kwargs):
        if False:
            return 10
        self.config = ConfigDict(kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        This method will use the output from backbone model to do any\n        downstream tasks. Receive The output from backbone model.\n\n        Returns (Dict[str, Any]): The output from downstream task.\n        '
        pass

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        compute loss for head during the finetuning.\n\n        Returns (Dict[str, Any]): The loss dict\n        '
        pass