"""An AdaNet interface for model search."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Sequence
from adanet.experimental.controllers.controller import Controller
from adanet.experimental.schedulers.in_process_scheduler import InProcessScheduler
from adanet.experimental.schedulers.scheduler import Scheduler
import tensorflow.compat.v2 as tf

class ModelSearch(object):
    """An AutoML pipeline manager."""

    def __init__(self, controller: Controller, scheduler: Scheduler=InProcessScheduler()):
        if False:
            i = 10
            return i + 15
        'Initializes a ModelSearch.\n\n    Args:\n      controller: A `Controller` instance.\n      scheduler: A `Scheduler` instance.\n    '
        self._controller = controller
        self._scheduler = scheduler

    def run(self):
        if False:
            i = 10
            return i + 15
        'Executes the training workflow to generate models.'
        self._scheduler.schedule(self._controller.work_units())

    def get_best_models(self, num_models) -> Sequence[tf.keras.Model]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the top models from the run.'
        return self._controller.get_best_models(num_models)