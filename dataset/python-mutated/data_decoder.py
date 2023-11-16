"""Interface for data decoders.

Data decoders decode the input data and return a dictionary of tensors keyed by
the entries in core.reader.Fields.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABCMeta
from abc import abstractmethod
import six

class DataDecoder(six.with_metaclass(ABCMeta, object)):
    """Interface for data decoders."""

    @abstractmethod
    def decode(self, data):
        if False:
            print('Hello World!')
        'Return a single image and associated labels.\n\n    Args:\n      data: a string tensor holding a serialized protocol buffer corresponding\n        to data for a single image.\n\n    Returns:\n      tensor_dict: a dictionary containing tensors. Possible keys are defined in\n          reader.Fields.\n    '
        pass