from st2common.metrics.base import BaseMetricsDriver
__all__ = ['NoopDriver']

class NoopDriver(BaseMetricsDriver):
    """
    Dummy implementation of BaseMetricsDriver
    """

    def __init__(self, *_args, **_kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass