"""Abstract base class for evaluation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc

class Scorer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._updated = False
        self._cached_results = {}

    @abc.abstractmethod
    def update(self, examples, predictions, loss):
        if False:
            for i in range(10):
                print('nop')
        self._updated = True

    @abc.abstractmethod
    def get_loss(self):
        if False:
            while True:
                i = 10
        pass

    @abc.abstractmethod
    def _get_results(self):
        if False:
            print('Hello World!')
        return []

    def get_results(self, prefix=''):
        if False:
            print('Hello World!')
        results = self._get_results() if self._updated else self._cached_results
        self._cached_results = results
        self._updated = False
        return [(prefix + k, v) for (k, v) in results]

    def results_str(self):
        if False:
            i = 10
            return i + 15
        return ' - '.join(['{:}: {:.2f}'.format(k, v) for (k, v) in self.get_results()])