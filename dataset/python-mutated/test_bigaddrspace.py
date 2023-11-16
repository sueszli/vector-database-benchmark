"""
These tests are meant to exercise that requests to create objects bigger
than what the address space allows are properly met with an OverflowError
(rather than crash weirdly).

Primarily, this means 32-bit builds with at least 2 GiB of available memory.
You need to pass the -M option to regrtest (e.g. "-M 2.1G") for tests to
be enabled.
"""
from test import support
from test.support import bigaddrspacetest, MAX_Py_ssize_t
import unittest
import operator
import sys

class BytesTest(unittest.TestCase):

    @bigaddrspacetest
    def test_concat(self):
        if False:
            while True:
                i = 10
        try:
            x = b'x' * (MAX_Py_ssize_t - 128)
            self.assertRaises(OverflowError, operator.add, x, b'x' * 128)
        finally:
            x = None

    @bigaddrspacetest
    def test_optimized_concat(self):
        if False:
            i = 10
            return i + 15
        try:
            x = b'x' * (MAX_Py_ssize_t - 128)
            with self.assertRaises(OverflowError) as cm:
                x = x + b'x' * 128
            with self.assertRaises(OverflowError) as cm:
                x += b'x' * 128
        finally:
            x = None

    @bigaddrspacetest
    def test_repeat(self):
        if False:
            return 10
        try:
            x = b'x' * (MAX_Py_ssize_t - 128)
            self.assertRaises(OverflowError, operator.mul, x, 128)
        finally:
            x = None

class StrTest(unittest.TestCase):
    unicodesize = 4

    @bigaddrspacetest
    def test_concat(self):
        if False:
            print('Hello World!')
        try:
            x = 'x' * int(MAX_Py_ssize_t // (1.1 * self.unicodesize))
            self.assertRaises(MemoryError, operator.add, x, x)
        finally:
            x = None

    @bigaddrspacetest
    def test_optimized_concat(self):
        if False:
            return 10
        try:
            x = 'x' * int(MAX_Py_ssize_t // (1.1 * self.unicodesize))
            with self.assertRaises(MemoryError) as cm:
                x = x + x
            with self.assertRaises(MemoryError) as cm:
                x += x
        finally:
            x = None

    @bigaddrspacetest
    def test_repeat(self):
        if False:
            while True:
                i = 10
        try:
            x = 'x' * int(MAX_Py_ssize_t // (1.1 * self.unicodesize))
            self.assertRaises(MemoryError, operator.mul, x, 2)
        finally:
            x = None
if __name__ == '__main__':
    if len(sys.argv) > 1:
        support.set_memlimit(sys.argv[1])
    unittest.main()