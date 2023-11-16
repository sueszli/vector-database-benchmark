"""Library of supported test cases.

All tests should conform to the NotebookTestCase abstract class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
from abc import ABC
from abc import abstractmethod

class NotebookTestCase(ABC):

    @abstractmethod
    def __init__(self, setup):
        if False:
            while True:
                i = 10
        'Construct a NotebookTestCase.\n\n    Args:\n      setup: arbitrary JSON-serializable object specified by test spec\n    '
        pass

    @abstractmethod
    def check(self, cell):
        if False:
            for i in range(10):
                print('nop')
        'Check correctness against single Jupyter cell.\n\n    Args:\n      cell: JSON representation of single cell.\n\n    Returns None if test succeeds, raise exception if test fails.\n    '
        pass

class RegexMatch(NotebookTestCase):
    """Checks if given string exists anywhere in the cell."""

    def __init__(self, setup):
        if False:
            print('Hello World!')
        self.regex = re.compile(setup)

    def check(self, cell):
        if False:
            print('Hello World!')
        if not self.regex.search(str(cell)):
            raise Exception('Could not find {} in {}'.format(self.regex.pattern, cell))