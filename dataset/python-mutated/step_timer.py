from collections import deque, defaultdict
from functools import wraps
from types import GeneratorType
from typing import Callable
import numpy as np
import time
from ditk import logging
from ding.framework import task

class StepTimer:

    def __init__(self, print_per_step: int=1, smooth_window: int=10) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Print time cost of each step (execute one middleware).\n        Arguments:\n            - print_per_step (:obj:`int`): Print each N step.\n            - smooth_window (:obj:`int`): The window size to smooth the mean.\n        '
        self.print_per_step = print_per_step
        self.records = defaultdict(lambda : deque(maxlen=print_per_step * smooth_window))

    def __call__(self, fn: Callable) -> Callable:
        if False:
            return 10
        step_name = getattr(fn, '__name__', type(fn).__name__)

        @wraps(fn)
        def executor(ctx):
            if False:
                while True:
                    i = 10
            start_time = time.time()
            time_cost = 0
            g = fn(ctx)
            if isinstance(g, GeneratorType):
                try:
                    next(g)
                except StopIteration:
                    pass
                time_cost = time.time() - start_time
                yield
                start_time = time.time()
                try:
                    next(g)
                except StopIteration:
                    pass
                time_cost += time.time() - start_time
            else:
                time_cost = time.time() - start_time
            self.records[step_name].append(time_cost)
            if ctx.total_step % self.print_per_step == 0:
                logging.info('[Step Timer][Node:{:>2}] {}: Cost: {:.2f}ms, Mean: {:.2f}ms'.format(task.router.node_id or 0, step_name, time_cost * 1000, np.mean(self.records[step_name]) * 1000))
        return executor