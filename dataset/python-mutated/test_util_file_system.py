from __future__ import absolute_import
import os
import os.path
import unittest2
from st2common.util.file_system import get_file_list
CURRENT_DIR = os.path.dirname(__file__)
ST2TESTS_DIR = os.path.join(CURRENT_DIR, '../../../st2tests/st2tests')

class FileSystemUtilsTestCase(unittest2.TestCase):

    def test_get_file_list(self):
        if False:
            for i in range(10):
                print('nop')
        directory = os.path.join(ST2TESTS_DIR, 'policies')
        expected = ['BUILD', 'mock_exception.py', 'concurrency.py', '__init__.py', 'meta/BUILD', 'meta/mock_exception.yaml', 'meta/concurrency.yaml', 'meta/__init__.py']
        result = get_file_list(directory=directory, exclude_patterns=['*.pyc'])
        self.assertItemsEqual(expected, result)
        expected = ['mock_exception.py', 'concurrency.py', '__init__.py', 'meta/__init__.py']
        result = get_file_list(directory=directory, exclude_patterns=['*.pyc', '*.yaml', '*BUILD'])
        self.assertItemsEqual(expected, result)