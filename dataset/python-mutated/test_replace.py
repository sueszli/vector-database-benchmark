"""
Created on Aug 5, 2011

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
from ...asttools.mutators.replace_mutator import replace_nodes
import ast
from ...asttools.tests import assert_ast_eq

class Test(unittest.TestCase):

    def test_replace_name(self):
        if False:
            while True:
                i = 10
        root = ast.parse('a = 1')
        name_a = root.body[0].targets[0]
        name_b = ast.Name(id='b', ctx=ast.Store())
        replace_nodes(root, name_a, name_b)
        expected = ast.parse('b = 1')
        assert_ast_eq(self, root, expected)

    def test_replace_non_existent(self):
        if False:
            print('Hello World!')
        root = ast.parse('a = 1')
        name_a = root.body[0].targets[0]
        name_b = ast.Name(id='b', ctx=ast.Store())
        replace_nodes(root, name_b, name_a)
        expected = ast.parse('a = 1')
        assert_ast_eq(self, root, expected)
if __name__ == '__main__':
    unittest.main()