"""Base classes for task-specific modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc

class SupervisedModule(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        if False:
            return 10
        self.supervised_loss = NotImplemented
        self.probs = NotImplemented
        self.preds = NotImplemented

    @abc.abstractmethod
    def update_feed_dict(self, feed, mb):
        if False:
            for i in range(10):
                print('nop')
        pass

class SemiSupervisedModule(SupervisedModule):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        if False:
            while True:
                i = 10
        super(SemiSupervisedModule, self).__init__()
        self.unsupervised_loss = NotImplemented