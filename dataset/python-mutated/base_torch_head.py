from typing import Any, Dict
import torch
from modelscope.models.base.base_head import Head
from modelscope.utils.logger import get_logger
logger = get_logger()

class TorchHead(Head, torch.nn.Module):
    """ Base head interface for pytorch

    """

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        torch.nn.Module.__init__(self)

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        if False:
            return 10
        raise NotImplementedError

    def compute_loss(self, *args, **kwargs) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        raise NotImplementedError