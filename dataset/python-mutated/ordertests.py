"""
Tests for handling of trial's --order option.
"""
from twisted.trial import unittest

class FooTest(unittest.TestCase):
    """
    Used to make assertions about the order its tests will be run in.
    """

    def test_first(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_second(self) -> None:
        if False:
            print('Hello World!')
        pass

    def test_third(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_fourth(self) -> None:
        if False:
            return 10
        pass

class BazTest(unittest.TestCase):
    """
    Used to make assertions about the order the test cases in this module are
    run in.
    """

    def test_baz(self) -> None:
        if False:
            return 10
        pass

class BarTest(unittest.TestCase):
    """
    Used to make assertions about the order the test cases in this module are
    run in.
    """

    def test_bar(self) -> None:
        if False:
            return 10
        pass