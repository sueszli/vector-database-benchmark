"""Interface for data parsers.

Data parser parses input data and returns a dictionary of numpy arrays
keyed by the entries in standard_fields.py. Since the parser parses records
to numpy arrays (materialized tensors) directly, it is used to read data for
evaluation/visualization; to parse the data during training, DataDecoder should
be used.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABCMeta
from abc import abstractmethod
import six

class DataToNumpyParser(six.with_metaclass(ABCMeta, object)):
    """Abstract interface for data parser that produces numpy arrays."""

    @abstractmethod
    def parse(self, input_data):
        if False:
            while True:
                i = 10
        'Parses input and returns a numpy array or a dictionary of numpy arrays.\n\n    Args:\n      input_data: an input data\n\n    Returns:\n      A numpy array or a dictionary of numpy arrays or None, if input\n      cannot be parsed.\n    '
        pass