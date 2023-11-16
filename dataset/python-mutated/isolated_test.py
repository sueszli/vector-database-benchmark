"""
Test isolated test. Isolated tests are run each using a new instance
of Python interpreter. They also implement some unique features for
our use case. See main_test.py for some real tests.
"""
import unittest
import _test_runner
from os.path import basename
g_count = 0

class IsolatedTest1(unittest.TestCase):

    def test_isolated1(self):
        if False:
            return 10
        global g_count
        g_count += 1
        self.assertEqual(g_count, 1)

    def test_isolated2(self):
        if False:
            print('Hello World!')
        global g_count
        g_count += 1
        self.assertEqual(g_count, 2)

class IsolatedTest2(unittest.TestCase):

    def test_isolated3(self):
        if False:
            for i in range(10):
                print('nop')
        global g_count
        g_count += 1
        self.assertEqual(g_count, 1)
if __name__ == '__main__':
    _test_runner.main(basename(__file__))