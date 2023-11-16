"""
Created on Aug 5, 2011

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
import ast
from ...asttools.visitors.symbol_visitor import get_symbols

class Test(unittest.TestCase):

    def assertHasSymbols(self, codestring, expected_symbols, ctxts=(ast.Load, ast.Store)):
        if False:
            while True:
                i = 10
        root = ast.parse(codestring)
        symbols = get_symbols(root, ctxts)
        self.assertEqual(symbols, expected_symbols)

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        self.assertHasSymbols('a', {'a'})

    def test_load(self):
        if False:
            i = 10
            return i + 15
        self.assertHasSymbols('a', {'a'}, ast.Load)
        self.assertHasSymbols('a', set(), ast.Store)

    def test_store(self):
        if False:
            print('Hello World!')
        self.assertHasSymbols('a = 1', {'a'}, ast.Store)
        self.assertHasSymbols('a = 1', set(), ast.Load)

    def test_store_item(self):
        if False:
            i = 10
            return i + 15
        self.assertHasSymbols('a[:] = 1', {'a'}, ast.Load)
        self.assertHasSymbols('a[:] = 1', set(), ast.Store)

    def test_store_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertHasSymbols('a.b = 1', {'a'}, ast.Load)
        self.assertHasSymbols('a.b = 1', set(), ast.Store)

    def test_for(self):
        if False:
            while True:
                i = 10
        self.assertHasSymbols('for i in x:\n    a.b = 1', {'a', 'x'}, ast.Load)
        self.assertHasSymbols('for i in x:\n    a.b = 1', {'i'}, ast.Store)
if __name__ == '__main__':
    unittest.main()