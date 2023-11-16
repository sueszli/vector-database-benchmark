from __future__ import absolute_import
from st2common.models.db.execution_queue import EXECUTION_QUEUE_ACCESS
from st2common.persistence import base as persistence
__all__ = ['ActionExecutionSchedulingQueue']

class ActionExecutionSchedulingQueue(persistence.Access):
    impl = EXECUTION_QUEUE_ACCESS
    publisher = None

    @classmethod
    def _get_impl(cls):
        if False:
            print('Hello World!')
        return cls.impl