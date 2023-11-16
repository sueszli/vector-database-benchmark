"""
Created on Aug 5, 2011

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
import ast
from ...asttools.mutators.remove_trivial import remove_trivial
from ...asttools.tests import assert_ast_eq, skip_networkx
from ...asttools.visitors.graph_visitor import GraphGen

def simple_case(self, toremove, expected):
    if False:
        return 10
    root = ast.parse(toremove)
    remove_trivial(root)
    expected_root = ast.parse(expected)
    assert_ast_eq(self, root, expected_root)

@skip_networkx
class Test(unittest.TestCase):

    def assertRemoved(self, toremove, expected):
        if False:
            print('Hello World!')
        root = ast.parse(toremove)
        remove_trivial(root)
        expected = ast.parse(expected)
        assert_ast_eq(self, root, expected)

    def test_single(self):
        if False:
            while True:
                i = 10
        simple_case(self, 'a = 1', 'a = 1')

    def test_empty(self):
        if False:
            i = 10
            return i + 15
        simple_case(self, '', '')

    def test_simple(self):
        if False:
            print('Hello World!')
        simple_case(self, 'a = 1; a = 2', 'pass; a = 2')

    def test_multi(self):
        if False:
            while True:
                i = 10
        simple_case(self, 'a = 1; a = 2; a = 3', 'pass; pass; a = 3')

    def test_apart(self):
        if False:
            while True:
                i = 10
        simple_case(self, 'a = 1; b = 1; a = 2', 'pass; b = 1; a = 2')

    def test_if(self):
        if False:
            print('Hello World!')
        simple_case(self, 'a = 1\nif x: a = 2', 'a = 1\nif x: a = 2')

    def test_if2(self):
        if False:
            while True:
                i = 10
        simple_case(self, 'if x: a = 2\na = 1', 'if x: a = 2\na = 1')

    def test_if_else(self):
        if False:
            return 10
        simple_case(self, 'a = 1\nif x: a = 2\nelse: a = 3', 'pass\nif x: a = 2\nelse: a = 3')

    def test_if_else2(self):
        if False:
            while True:
                i = 10
        simple_case(self, 'if x: a = 2\nelse: a = 3\na = 1', 'if x: pass\nelse: pass\na = 1')

    def test_for(self):
        if False:
            i = 10
            return i + 15
        simple_case(self, 'a = 1\nfor x in y: a = 2', 'a = 1\nfor x in y: a = 2')

    def test_for_else(self):
        if False:
            print('Hello World!')
        simple_case(self, 'a = 1\nfor x in y: a = 2\nelse: a = 3', 'pass\nfor x in y: a = 2\nelse: a = 3')

    def test_for_else_break(self):
        if False:
            print('Hello World!')
        simple_case(self, 'a = 1\nfor x in y:\n    break\n    a = 2\nelse: a = 3', 'a = 1\nfor x in y:\n    break\n    a = 2\nelse: a = 3')

    def test_for_else_conti(self):
        if False:
            while True:
                i = 10
        simple_case(self, 'a = 1\nfor x in y:\n    continue\n    a = 2\nelse: a = 3', 'a = 1\nfor x in y:\n    continue\n    a = 2\nelse: a = 3')

    def test_while(self):
        if False:
            for i in range(10):
                print('nop')
        simple_case(self, 'a = 1\nwhile x: a = 2', 'a = 1\nwhile x: a = 2')

    def test_while_else(self):
        if False:
            return 10
        simple_case(self, 'a = 1\nwhile x: a = 2\nelse: a = 3', 'pass\nwhile x: a = 2\nelse: a = 3')
if __name__ == '__main__':
    unittest.main()