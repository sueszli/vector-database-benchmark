"""
Created on Aug 2, 2011

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
import ast
from ...asttools.visitors.graph_visitor import GraphGen
from ...asttools.visitors.graph_visitor import DiGraph
from ...asttools.tests import AllTypesTested, skip_networkx
tested = AllTypesTested()

def binop_method(op):
    if False:
        return 10

    def test_binop(self):
        if False:
            for i in range(10):
                print('nop')
        source = 'c = a %s b' % (op,)
        self.assertDepends(source, {('c', 'a'), ('c', 'b')}, {'a', 'b'}, {'c'})
    return test_binop

def unarynop_method(op):
    if False:
        while True:
            i = 10

    def test_unaryop(self):
        if False:
            while True:
                i = 10
        source = 'c = %s b' % (op,)
        self.assertDepends(source, {('c', 'b')}, {'b'}, {'c'})
    return test_unaryop

@skip_networkx
class Test(unittest.TestCase):

    def assertDepends(self, source, edges, undefined=None, modified=None):
        if False:
            while True:
                i = 10
        mod = ast.parse(source)
        gen = GraphGen(call_deps=True)
        gen.visit(mod)
        self.assertSetEqual(set(gen.graph.edges()), edges)
        if undefined is not None:
            self.assertSetEqual(set(gen.undefined), undefined)
        if modified is not None:
            self.assertSetEqual(set(gen.modified), modified)
        tested.update(mod)
        return gen

    def test_name(self):
        if False:
            for i in range(10):
                print('nop')
        source = 'a'
        self.assertDepends(source, set())

    def test_assign(self):
        if False:
            i = 10
            return i + 15
        source = 'a = b'
        self.assertDepends(source, {('a', 'b')}, {'b'}, {'a'})

    def test_assign_tuple(self):
        if False:
            i = 10
            return i + 15
        source = '(a, c) = b'
        self.assertDepends(source, {('a', 'b'), ('c', 'b')}, {'b'}, {'a', 'c'})

    def test_assign_multi(self):
        if False:
            print('Hello World!')
        source = 'a = b  = c'
        self.assertDepends(source, {('a', 'c'), ('b', 'c')}, {'c'}, {'a', 'b'})

    def test_assign_attr(self):
        if False:
            return 10
        source = 'a.x = b'
        self.assertDepends(source, {('a', 'b')}, {'b', 'a'}, {'a'})

    def test_attr_assign(self):
        if False:
            while True:
                i = 10
        source = 'a = b.x'
        self.assertDepends(source, {('a', 'b')}, {'b'}, {'a'})

    def test_subscr(self):
        if False:
            return 10
        source = 'a[:] = b[:]'
        self.assertDepends(source, {('a', 'b')}, {'a', 'b'}, {'a'})

    def test_subscr_value(self):
        if False:
            return 10
        source = 'a = b[c]'
        self.assertDepends(source, {('a', 'b'), ('a', 'c')}, {'b', 'c'}, {'a'})

    def test_subscr_lvalue(self):
        if False:
            return 10
        source = 'a[c] = b'
        self.assertDepends(source, {('a', 'b'), ('a', 'c')}, {'a', 'b', 'c'}, {'a'})

    def test_subscr_attr(self):
        if False:
            for i in range(10):
                print('nop')
        source = 'a[:] = b[:].b'
        self.assertDepends(source, {('a', 'b')}, {'a', 'b'}, {'a'})

    def test_import(self):
        if False:
            return 10
        source = 'import foo; foo.a = b'
        self.assertDepends(source, {('foo', 'b')}, {'b'}, {'foo'})

    def test_import_from(self):
        if False:
            for i in range(10):
                print('nop')
        source = 'from bar import foo; foo.a = b'
        self.assertDepends(source, {('foo', 'b')}, {'b'}, {'foo'})

    def test_import_as(self):
        if False:
            i = 10
            return i + 15
        source = 'import bar as foo; foo.a = b'
        self.assertDepends(source, {('foo', 'b')}, {'b'}, {'foo'})

    def test_import_from_as(self):
        if False:
            return 10
        source = 'from bar import baz as foo; foo.a = b'
        self.assertDepends(source, {('foo', 'b')}, {'b'}, {'foo'})

    def test_augment_assign(self):
        if False:
            print('Hello World!')
        source = 'a += b'
        self.assertDepends(source, {('a', 'b'), ('a', 'a')}, {'b'}, {'a'})
    test_add = binop_method('+')
    test_sub = binop_method('-')
    test_pow = binop_method('**')
    test_eq = binop_method('==')
    test_ne = binop_method('!=')
    test_rshift = binop_method('>>')
    test_lshift = binop_method('<<')
    test_mult = binop_method('*')
    test_mod = binop_method('%')
    test_div = binop_method('/')
    test_floordiv = binop_method('//')
    test_bitxor = binop_method('^')
    test_lt = binop_method('<')
    test_gt = binop_method('>')
    test_lte = binop_method('<=')
    test_gte = binop_method('>=')
    test_in = binop_method('in')
    test_not_in = binop_method('not in')
    test_is = binop_method('is')
    test_is_not = binop_method('is not')
    test_bit_or = binop_method('|')
    test_bit_and = binop_method('&')
    test_or = binop_method('or')
    test_and = binop_method('and')
    test_not = unarynop_method('not')
    test_uadd = unarynop_method('+')
    test_usub = unarynop_method('-')
    test_invert = unarynop_method('~')

    def test_call(self):
        if False:
            print('Hello World!')
        source = 'foo(a)'
        self.assertDepends(source, {('foo', 'a'), ('a', 'foo')}, {'a', 'foo'})

    def test_for(self):
        if False:
            return 10
        source = 'for i in a:\n    b'
        self.assertDepends(source, {('i', 'a'), ('b', 'a')}, {'a', 'b'}, {'i'})

    def test_for2(self):
        if False:
            return 10
        source = 'for i in a:\n    x += b[i]'
        self.assertDepends(source, {('i', 'a'), ('b', 'a'), ('x', 'a'), ('x', 'i'), ('x', 'b'), ('x', 'x')}, {'a', 'b'}, {'x', 'i'})

    def test_for_unpack(self):
        if False:
            while True:
                i = 10
        source = 'for i, j in a:\n    x += b[i]'
        self.assertDepends(source, {('i', 'a'), ('j', 'a'), ('b', 'a'), ('x', 'a'), ('x', 'i'), ('x', 'b'), ('x', 'x')}, {'a', 'b'}, {'x', 'i', 'j'})

    def test_dict(self):
        if False:
            return 10
        source = 'c = {a:b}'
        self.assertDepends(source, {('c', 'a'), ('c', 'b')}, {'a', 'b'}, {'c'})

    def test_list(self):
        if False:
            while True:
                i = 10
        source = 'c = [a,b]'
        self.assertDepends(source, {('c', 'a'), ('c', 'b')}, {'a', 'b'}, {'c'})

    def test_tuple(self):
        if False:
            i = 10
            return i + 15
        source = 'c = (a,b)'
        self.assertDepends(source, {('c', 'a'), ('c', 'b')}, {'a', 'b'}, {'c'})

    def test_set(self):
        if False:
            print('Hello World!')
        source = 'c = {a,b}'
        self.assertDepends(source, {('c', 'a'), ('c', 'b')}, {'a', 'b'}, {'c'})

    def test_if(self):
        if False:
            print('Hello World!')
        source = 'if a: b'
        self.assertDepends(source, {('b', 'a')}, {'a', 'b'}, set())

    def test_if_else(self):
        if False:
            print('Hello World!')
        source = 'if a: b\nelse: c'
        self.assertDepends(source, {('b', 'a'), ('c', 'a')}, {'a', 'b', 'c'}, set())

    def test_if_elif_else(self):
        if False:
            i = 10
            return i + 15
        source = 'if a: b\nelif x: c\nelse: d'
        self.assertDepends(source, {('b', 'a'), ('c', 'x'), ('c', 'a'), ('d', 'a'), ('d', 'x'), ('x', 'a')}, {'a', 'b', 'c', 'd', 'x'}, set())

    def test_if_expr(self):
        if False:
            while True:
                i = 10
        source = 'd = b if a else c'
        self.assertDepends(source, {('d', 'a'), ('d', 'b'), ('d', 'c')}, {'a', 'b', 'c'}, {'d'})

    def test_assert(self):
        if False:
            while True:
                i = 10
        source = 'assert a'
        self.assertDepends(source, set(), {'a'}, set())

    def test_with(self):
        if False:
            return 10
        source = 'with a as b: c'
        self.assertDepends(source, {('b', 'a'), ('c', 'a')}, {'a', 'c'}, {'b'})

    def test_while(self):
        if False:
            while True:
                i = 10
        source = 'while a: c'
        self.assertDepends(source, {('c', 'a')}, {'a', 'c'})

    def test_function_def(self):
        if False:
            for i in range(10):
                print('nop')
        source = 'a = 1\ndef foo(b):\n    return a + b\n'
        self.assertDepends(source, {('foo', 'a')})

    def test_lambda(self):
        if False:
            print('Hello World!')
        source = 'a = 1\nfoo = lambda b:  a + b\n'
        self.assertDepends(source, {('foo', 'a')})

    def test_list_comp(self):
        if False:
            return 10
        source = 'a = [b for b in c]'
        self.assertDepends(source, {('a', 'c')})

    def test_dict_comp(self):
        if False:
            i = 10
            return i + 15
        source = 'a = {b:d for b,d in c}'
        self.assertDepends(source, {('a', 'c')})

    def test_set_comp(self):
        if False:
            return 10
        source = 'a = {b for b in c}'
        self.assertDepends(source, {('a', 'c')})

    def test_try_except(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\ntry:\n    a\nexcept b:\n    c\n        '
        self.assertDepends(source, {('c', 'a'), ('c', 'b')})

    def test_try_except_else(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\ntry:\n    a\nexcept b:\n    c\nelse:\n    d\n        '
        self.assertDepends(source, {('c', 'a'), ('c', 'b'), ('d', 'a')})

    def test_try_finally(self):
        if False:
            return 10
        source = '\ntry:\n    a\nexcept b:\n    c\nfinally:\n    d\n        '
        self.assertDepends(source, {('c', 'a'), ('c', 'b'), ('d', 'a'), ('d', 'b'), ('d', 'c')})
if __name__ == '__main__':
    unittest.main(exit=False)
    print(tested.tested())