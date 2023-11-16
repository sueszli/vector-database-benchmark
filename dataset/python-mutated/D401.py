"""This module docstring does not need to be written in imperative mood."""
from functools import cached_property
from gi.repository import GObject

def bad_liouiwnlkjl():
    if False:
        while True:
            i = 10
    'Returns foo.'

def bad_sdgfsdg23245():
    if False:
        while True:
            i = 10
    'Constructor for a foo.'

def bad_sdgfsdg23245777():
    if False:
        return 10
    '\n\n    Constructor for a boa.\n\n    '

def bad_run_something():
    if False:
        print('Hello World!')
    'Runs something'

    def bad_nested():
        if False:
            return 10
        'Runs other things, nested'
    bad_nested()

def multi_line():
    if False:
        print('Hello World!')
    'Writes a logical line that\n    extends to two physical lines.\n    '

def good_run_something():
    if False:
        while True:
            i = 10
    'Run away.'

    def good_nested():
        if False:
            print('Hello World!')
        'Run to the hills.'
    good_nested()

def good_construct():
    if False:
        while True:
            i = 10
    'Construct a beautiful house.'

def good_multi_line():
    if False:
        print('Hello World!')
    'Write a logical line that\n    extends to two physical lines.\n    '
good_top_level_var = False
'This top level assignment attribute docstring does not need to be written in imperative mood.'

class Thingy:
    """This class docstring does not need to be written in imperative mood."""
    _beep = 'boop'
    'This class attribute docstring does not need to be written in imperative mood.'

    def bad_method(self):
        if False:
            print('Hello World!')
        'This method docstring should be written in imperative mood.'

    @property
    def good_property(self):
        if False:
            return 10
        'This property method docstring does not need to be written in imperative mood.'
        return self._beep

    @GObject.Property
    def good_custom_property(self):
        if False:
            while True:
                i = 10
        'This property method docstring does not need to be written in imperative mood.'
        return self._beep

    @cached_property
    def good_cached_property(self):
        if False:
            print('Hello World!')
        'This property method docstring does not need to be written in imperative mood.'
        return 42 * 42

    class NestedThingy:
        """This nested class docstring does not need to be written in imperative mood."""

def test_something():
    if False:
        i = 10
        return i + 15
    'This test function does not need to be written in imperative mood.\n\n    pydocstyle\'s rationale:\n    We exclude tests from the imperative mood check, because to phrase\n    their docstring in the imperative mood, they would have to start with\n    a highly redundant "Test that ..."\n    '

def runTest():
    if False:
        i = 10
        return i + 15
    'This test function does not need to be written in imperative mood, either.'