from typing import Dict
from modelscope.metainfo import Metrics
from modelscope.utils.registry import default_group
from .base import Metric
from .builder import METRICS

@METRICS.register_module(group_key=default_group, module_name=Metrics.prediction_saving_wrapper)
class PredictionSavingWrapper(Metric):
    """The wrapper to save predictions to file.
    Args:
        saving_fn: The saving_fn used to save predictions to files.
    """

    def __init__(self, saving_fn, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.saving_fn = saving_fn

    def add(self, outputs: Dict, inputs: Dict):
        if False:
            print('Hello World!')
        self.saving_fn(inputs, outputs)

    def evaluate(self):
        if False:
            for i in range(10):
                print('nop')
        return {}

    def merge(self, other: 'PredictionSavingWrapper'):
        if False:
            while True:
                i = 10
        pass

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        pass

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        pass