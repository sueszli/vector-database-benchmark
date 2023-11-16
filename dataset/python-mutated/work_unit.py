"""A work unit for an AdaNet scheduler."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc

class WorkUnit(abc.ABC):

    @abc.abstractproperty
    def execute(self):
        if False:
            return 10
        pass