"""
Abstract class for task history.
Currently the only subclass is :py:class:`~luigi.db_task_history.DbTaskHistory`.
"""
import abc
import logging
logger = logging.getLogger('luigi-interface')

class StoredTask:
    """
    Interface for methods on TaskHistory
    """

    def __init__(self, task, status, host=None):
        if False:
            return 10
        self._task = task
        self.status = status
        self.record_id = None
        self.host = host

    @property
    def task_family(self):
        if False:
            while True:
                i = 10
        return self._task.family

    @property
    def parameters(self):
        if False:
            for i in range(10):
                print('nop')
        return self._task.params

class TaskHistory(metaclass=abc.ABCMeta):
    """
    Abstract Base Class for updating the run history of a task
    """

    @abc.abstractmethod
    def task_scheduled(self, task):
        if False:
            i = 10
            return i + 15
        pass

    @abc.abstractmethod
    def task_finished(self, task, successful):
        if False:
            print('Hello World!')
        pass

    @abc.abstractmethod
    def task_started(self, task, worker_host):
        if False:
            for i in range(10):
                print('nop')
        pass

class NopHistory(TaskHistory):

    def task_scheduled(self, task):
        if False:
            print('Hello World!')
        pass

    def task_finished(self, task, successful):
        if False:
            print('Hello World!')
        pass

    def task_started(self, task, worker_host):
        if False:
            while True:
                i = 10
        pass