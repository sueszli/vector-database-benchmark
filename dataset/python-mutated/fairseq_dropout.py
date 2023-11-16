import logging
from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F
logger = logging.getLogger(__name__)

class FairseqDropout(nn.Module):

    def __init__(self, p, module_name=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool=False):
        if False:
            while True:
                i = 10
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(self, name: str, retain_dropout: bool=False, retain_dropout_modules: Optional[List[str]]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning('Cannot enable dropout during inference for module {} because module_name was not set'.format(name))
            elif retain_dropout_modules is None or self.module_name in retain_dropout_modules:
                logger.info('Enabling dropout during inference for module: {}'.format(name))
                self.apply_during_inference = True
            else:
                logger.info('Disabling dropout for module: {}'.format(name))