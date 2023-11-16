"""Tests that run all fixer modules over an input stream.

This has been broken out into its own test module because of its
running time.
"""
import os.path
import sys
import test.support
import unittest
from . import support

@test.support.requires_resource('cpu')
class Test_all(support.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.refactor = support.get_refactorer()

    def refactor_file(self, filepath):
        if False:
            for i in range(10):
                print('nop')
        if test.support.verbose:
            print(f'Refactor file: {filepath}')
        if os.path.basename(filepath) == 'infinite_recursion.py':
            with test.support.infinite_recursion(150):
                self.refactor.refactor_file(filepath)
        else:
            self.refactor.refactor_file(filepath)

    def test_all_project_files(self):
        if False:
            i = 10
            return i + 15
        for filepath in support.all_project_files():
            with self.subTest(filepath=filepath):
                self.refactor_file(filepath)
if __name__ == '__main__':
    unittest.main()