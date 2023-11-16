"""
Direct unit tests for L{twisted.trial.unittest.SynchronousTestCase} and
L{twisted.trial.unittest.TestCase}.
"""
from twisted.trial.unittest import SynchronousTestCase, TestCase

class TestCaseMixin:
    """
    L{TestCase} tests.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        '\n        Create a couple instances of C{MyTestCase}, each for the same test\n        method, to be used in the test methods of this class.\n        '
        self.first = self.MyTestCase('test_1')
        self.second = self.MyTestCase('test_1')

    def test_equality(self):
        if False:
            i = 10
            return i + 15
        '\n        In order for one test method to be runnable twice, two TestCase\n        instances with the same test method name must not compare as equal.\n        '
        self.assertTrue(self.first == self.first)
        self.assertTrue(self.first != self.second)
        self.assertFalse(self.first == self.second)

    def test_hashability(self):
        if False:
            return 10
        '\n        In order for one test method to be runnable twice, two TestCase\n        instances with the same test method name should not have the same\n        hash value.\n        '
        container = {}
        container[self.first] = None
        container[self.second] = None
        self.assertEqual(len(container), 2)

class SynchronousTestCaseTests(TestCaseMixin, SynchronousTestCase):

    class MyTestCase(SynchronousTestCase):
        """
        Some test methods which can be used to test behaviors of
        L{SynchronousTestCase}.
        """

        def test_1(self):
            if False:
                while True:
                    i = 10
            pass

class AsynchronousTestCaseTests(TestCaseMixin, SynchronousTestCase):

    class MyTestCase(TestCase):
        """
        Some test methods which can be used to test behaviors of
        L{TestCase}.
        """

        def test_1(self):
            if False:
                return 10
            pass