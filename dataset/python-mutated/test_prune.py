"""
Created on Aug 2, 2011

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from ...asttools.tests import AllTypesTested, assert_ast_eq
import unittest
from ...asttools.mutators.prune_mutator import PruneVisitor
import ast
from ...testing import py2only
tested = AllTypesTested()

class TestExclusive(unittest.TestCase):

    def assertPruned(self, source, pruned, symbols):
        if False:
            return 10
        mutator = PruneVisitor(symbols=symbols, mode='exclusive')
        orig_ast = ast.parse(source)
        expected_ast = ast.parse(pruned)
        mutator.visit(orig_ast)
        assert_ast_eq(self, orig_ast, expected_ast)
        tested.update(orig_ast)

    def test_assign(self):
        if False:
            return 10
        source = 'a = b; c = d'
        pruned = 'a = b;'
        self.assertPruned(source, pruned, symbols=['c', 'd'])
        pruned2 = 'c = d'
        self.assertPruned(source, pruned2, symbols=['a', 'b'])
        pruned = 'a = b; c = d'
        self.assertPruned(source, pruned, symbols=['c'])
        pruned2 = 'a = b; c = d'
        self.assertPruned(source, pruned2, symbols=['b'])

    def test_binop(self):
        if False:
            while True:
                i = 10
        source = 'a + b; c + d'
        pruned = 'a + b'
        self.assertPruned(source, pruned, symbols=['c', 'd'])

    def test_unaryop(self):
        if False:
            while True:
                i = 10
        source = '+b; -c'
        pruned = '+b'
        self.assertPruned(source, pruned, symbols=['c'])

    def test_for(self):
        if False:
            return 10
        source = 'for i in j: k'
        pruned = 'for i in j: pass'
        self.assertPruned(source, pruned, symbols=['k'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['k', 'i', 'j'])

    def test_for_else(self):
        if False:
            print('Hello World!')
        source = 'for i in j:\n    k\nelse:\n    l'
        pruned = 'for i in j:\n    k'
        self.assertPruned(source, pruned, symbols=['l'])
        pruned = 'for i in j:\n    pass\nelse:\n    l'
        self.assertPruned(source, pruned, symbols=['i', 'j', 'k'])

    def test_with_as(self):
        if False:
            return 10
        source = 'with a as b: c'
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'b', 'c'])
        pruned = 'with a as b: pass'
        self.assertPruned(source, pruned, symbols=['c'])

    def test_with(self):
        if False:
            while True:
                i = 10
        source = 'with a: c'
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'c'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['c'])

    def test_if(self):
        if False:
            while True:
                i = 10
        source = 'if a: b\nelse: c'
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'b', 'c'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['b', 'c'])
        pruned = 'if a: b'
        self.assertPruned(source, pruned, symbols=['c'])

    def test_if_expr(self):
        if False:
            return 10
        source = 'a = b if c else d'
        pruned = 'a = b if c else d'
        self.assertPruned(source, pruned, symbols=['b', 'c', 'd'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'b', 'c', 'd'])

    def test_while(self):
        if False:
            i = 10
            return i + 15
        source = 'while a: b'
        pruned = ''
        self.assertPruned(source, pruned, symbols=['b'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'b', 'c', 'd'])

    def test_import(self):
        if False:
            while True:
                i = 10
        source = 'import a'
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a'])
        source = 'import a, b'
        pruned = 'import a, b'
        self.assertPruned(source, pruned, symbols=['a'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'b'])

    def test_import_from(self):
        if False:
            i = 10
            return i + 15
        source = 'from a import b'
        pruned = 'from a import b'
        self.assertPruned(source, pruned, symbols=['a'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['b'])

    def test_try(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\ntry:\n    a\nexcept b as c:\n    d\n'
        pruned = '\n'
        self.assertPruned(source, pruned, symbols=['a', 'b', 'c', 'd'])
        pruned = '\ntry:\n    a\nexcept b as c:\n    pass\n'
        self.assertPruned(source, pruned, symbols=['d'])
        pruned = '\n'
        self.assertPruned(source, pruned, symbols=['a', 'd'])

    def test_try_else(self):
        if False:
            while True:
                i = 10
        source = '\ntry:\n    a\nexcept b as c:\n    d\nelse:\n    e\n'
        pruned = '\ntry:\n    pass\nexcept:\n    pass\nelse:\n    e\n\n'
        self.assertPruned(source, pruned, symbols=['a'])

    def test_try_finally(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\ntry:\n    a\nexcept b as c:\n    d\nelse:\n    e\nfinally:\n    f\n'
        pruned = '\ntry:\n    pass\nexcept:\n    pass\nelse:\n    e\nfinally:\n    f\n\n'
        self.assertPruned(source, pruned, symbols=['a'])
        pruned = '\ntry:\n    pass\nexcept:\n    pass\nelse:\n    e\nfinally:\n    pass\n\n'
        self.assertPruned(source, pruned, symbols=['a', 'f'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'd', 'e', 'f'])

    @py2only
    def test_exec(self):
        if False:
            while True:
                i = 10
        source = 'exec a'
        pruned = 'exec a'
        self.assertPruned(source, pruned, symbols=['a'])

    def test_attr(self):
        if False:
            i = 10
            return i + 15
        pass

class TestInclusive(unittest.TestCase):

    def assertPruned(self, source, pruned, symbols):
        if False:
            print('Hello World!')
        mutator = PruneVisitor(symbols=symbols, mode='inclusive')
        orig_ast = ast.parse(source)
        expected_ast = ast.parse(pruned)
        mutator.visit(orig_ast)
        assert_ast_eq(self, orig_ast, expected_ast)
        tested.update(orig_ast)

    def test_assign(self):
        if False:
            i = 10
            return i + 15
        source = 'a = b; c = d'
        pruned = 'a = b;'
        self.assertPruned(source, pruned, symbols=['c', 'd'])
        pruned2 = 'c = d'
        self.assertPruned(source, pruned2, symbols=['a', 'b'])
        pruned = 'a = b'
        self.assertPruned(source, pruned, symbols=['c'])
        pruned2 = 'c = d'
        self.assertPruned(source, pruned2, symbols=['b'])

    def test_binop(self):
        if False:
            return 10
        source = 'a + b; c + d'
        pruned = 'a + b'
        self.assertPruned(source, pruned, symbols=['c', 'd'])

    def test_unaryop(self):
        if False:
            i = 10
            return i + 15
        source = '+b; -c'
        pruned = '+b'
        self.assertPruned(source, pruned, symbols=['c'])

    def test_for(self):
        if False:
            for i in range(10):
                print('nop')
        source = 'for i in j: k'
        pruned = 'for i in j: pass'
        self.assertPruned(source, pruned, symbols=['k'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['k', 'i', 'j'])

    def test_for_else(self):
        if False:
            while True:
                i = 10
        source = 'for i in j:\n    k\nelse:\n    l'
        pruned = 'for i in j:\n    k'
        self.assertPruned(source, pruned, symbols=['l'])
        pruned = 'for i in j:\n    pass\nelse:\n    l'
        self.assertPruned(source, pruned, symbols=['i', 'j', 'k'])

    def test_with_as(self):
        if False:
            return 10
        source = 'with a as b: c'
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'b', 'c'])
        pruned = 'with a as b: pass'
        self.assertPruned(source, pruned, symbols=['c'])

    def test_with(self):
        if False:
            print('Hello World!')
        source = 'with a: c'
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'c'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['c'])

    def test_if(self):
        if False:
            print('Hello World!')
        source = 'if a: b\nelse: c'
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'b', 'c'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['b', 'c'])
        pruned = 'if a: b'
        self.assertPruned(source, pruned, symbols=['c'])

    def test_if_expr(self):
        if False:
            print('Hello World!')
        source = 'a = b if c else d'
        pruned = ''
        self.assertPruned(source, pruned, symbols=['b', 'c', 'd'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'b', 'c', 'd'])

    def test_while(self):
        if False:
            return 10
        source = 'while a: b'
        pruned = ''
        self.assertPruned(source, pruned, symbols=['b'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'b', 'c', 'd'])

    def test_import(self):
        if False:
            while True:
                i = 10
        source = 'import a'
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a'])
        source = 'import a, b'
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'b'])

    def test_import_from(self):
        if False:
            i = 10
            return i + 15
        source = 'from a import b'
        pruned = 'from a import b'
        self.assertPruned(source, pruned, symbols=['a'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['b'])

    def test_try(self):
        if False:
            while True:
                i = 10
        source = '\ntry:\n    a\nexcept b as c:\n    d\n'
        pruned = '\n'
        self.assertPruned(source, pruned, symbols=['a', 'b', 'c', 'd'])
        pruned = '\ntry:\n    a\nexcept b as c:\n    pass\n'
        self.assertPruned(source, pruned, symbols=['d'])
        pruned = '\n'
        self.assertPruned(source, pruned, symbols=['a', 'd'])

    def test_try_else(self):
        if False:
            while True:
                i = 10
        source = '\ntry:\n    a\nexcept b as c:\n    d\nelse:\n    e\n'
        pruned = '\ntry:\n    pass\nexcept:\n    pass\nelse:\n    e\n\n'
        self.assertPruned(source, pruned, symbols=['a'])

    def test_try_finally(self):
        if False:
            i = 10
            return i + 15
        source = '\ntry:\n    a\nexcept b as c:\n    d\nelse:\n    e\nfinally:\n    f\n'
        pruned = '\ntry:\n    pass\nexcept:\n    pass\nelse:\n    e\nfinally:\n    f\n\n'
        self.assertPruned(source, pruned, symbols=['a'])
        pruned = '\ntry:\n    pass\nexcept:\n    pass\nelse:\n    e\nfinally:\n    pass\n\n'
        self.assertPruned(source, pruned, symbols=['a', 'f'])
        pruned = ''
        self.assertPruned(source, pruned, symbols=['a', 'd', 'e', 'f'])

    @py2only
    def test_exec(self):
        if False:
            print('Hello World!')
        source = 'exec a'
        pruned = 'exec a'
        self.assertPruned(source, pruned, symbols=['a'])

    def test_attr(self):
        if False:
            while True:
                i = 10
        pass
if __name__ == '__main__':
    unittest.main(exit=False)
    print(tested.tested())