import sys
import unittest
from numba import njit

@njit
def f0(a, b):
    if False:
        return 10
    return a + b

@njit
def f1(begin1, end1, begin2, end2):
    if False:
        while True:
            i = 10
    if begin1 > begin2:
        return f1(begin2, end2, begin1, end1)
    return end1 + 1 >= begin2

@njit
def f0_2(a, b):
    if False:
        i = 10
        return i + 15
    return a + b

@njit
def f1_2(begin1, end1, begin2, end2):
    if False:
        while True:
            i = 10
    if begin1 > begin2:
        return f1_2(begin2, end2, begin1, end1)
    return end1 + 1 >= begin2

class TestSysStdinAssignment(unittest.TestCase):

    def test_no_reassignment_of_stdout(self):
        if False:
            i = 10
            return i + 15
        '\n        https://github.com/numba/numba/issues/3027\n        Older versions of colorama break stdout/err when recursive functions\n        are compiled.\n\n        This test should work irrespective of colorama version, or indeed its\n        presence. If the version is too low, it should be disabled and the test\n        should work anyway, if it is a sufficiently high version or it is not\n        present, it should work anyway.\n        '
        originally = (sys.stdout, sys.stderr)
        try:
            sys.stdout = None
            f0(0, 1)
            self.assertEqual(sys.stdout, None)
            f1(0, 1, 2, 3)
            self.assertEqual(sys.stdout, None)
            sys.stderr = None
            f0_2(0, 1)
            self.assertEqual(sys.stderr, None)
            f1_2(0, 1, 2, 3)
            self.assertEqual(sys.stderr, None)
        finally:
            (sys.stdout, sys.stderr) = originally
        self.assertNotEqual(sys.stderr, None)
        self.assertNotEqual(sys.stdout, None)
if __name__ == '__main__':
    unittest.main()