from typing import Callable
import torch
from .time_helper_base import TimeWrapper

def get_cuda_time_wrapper() -> Callable[[], 'TimeWrapper']:
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Return the ``TimeWrapperCuda`` class, this wrapper aims to ensure compatibility in no cuda device\n\n    Returns:\n        - TimeWrapperCuda(:obj:`class`): See ``TimeWrapperCuda`` class\n\n    .. note::\n        Must use ``torch.cuda.synchronize()``, reference: <https://blog.csdn.net/u013548568/article/details/81368019>\n\n    '

    class TimeWrapperCuda(TimeWrapper):
        """
        Overview:
            A class method that inherit from ``TimeWrapper`` class

            Notes:
                Must use torch.cuda.synchronize(), reference: \\
                <https://blog.csdn.net/u013548568/article/details/81368019>

        Interface:
            ``start_time``, ``end_time``
        """
        start_record = torch.cuda.Event(enable_timing=True)
        end_record = torch.cuda.Event(enable_timing=True)

        @classmethod
        def start_time(cls):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Overview:\n                Implement and overide the ``start_time`` method in ``TimeWrapper`` class\n            '
            torch.cuda.synchronize()
            cls.start = cls.start_record.record()

        @classmethod
        def end_time(cls):
            if False:
                return 10
            '\n            Overview:\n                Implement and overide the end_time method in ``TimeWrapper`` class\n            Returns:\n                - time(:obj:`float`): The time between ``start_time`` and ``end_time``\n            '
            cls.end = cls.end_record.record()
            torch.cuda.synchronize()
            return cls.start_record.elapsed_time(cls.end_record) / 1000
    return TimeWrapperCuda