from __future__ import annotations
from ..basics import *
__all__ = ['MCDropoutCallback']

class MCDropoutCallback(Callback):

    def before_validate(self):
        if False:
            for i in range(10):
                print('nop')
        for m in [m for m in flatten_model(self.model) if 'dropout' in m.__class__.__name__.lower()]:
            m.train()

    def after_validate(self):
        if False:
            return 10
        for m in [m for m in flatten_model(self.model) if 'dropout' in m.__class__.__name__.lower()]:
            m.eval()