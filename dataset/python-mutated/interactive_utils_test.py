"""Tests for interactive_utils module."""
import unittest
from unittest.mock import patch
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive.testing.mock_ipython import mock_get_ipython
from apache_beam.utils.interactive_utils import is_in_ipython

def unavailable_ipython():
    if False:
        while True:
            i = 10
    raise ImportError('Module IPython is not found.')

def corrupted_ipython():
    if False:
        print('Hello World!')
    raise AttributeError('Module IPython does not contain get_ipython.')

@unittest.skipIf(not ie.current_env().is_interactive_ready, '[interactive] dependency is not installed.')
class IPythonTest(unittest.TestCase):

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    def test_is_in_ipython_when_in_ipython_kernel(self, kernel):
        if False:
            return 10
        self.assertTrue(is_in_ipython())

    @patch('IPython.get_ipython', new_callable=lambda : unavailable_ipython)
    def test_is_not_in_ipython_when_no_ipython_dep(self, unavailable):
        if False:
            i = 10
            return i + 15
        self.assertFalse(is_in_ipython())

    @patch('IPython.get_ipython', new_callable=lambda : corrupted_ipython)
    def test_is_not_ipython_when_ipython_errors_out(self, corrupted):
        if False:
            print('Hello World!')
        self.assertFalse(is_in_ipython())
if __name__ == '__main__':
    unittest.main()