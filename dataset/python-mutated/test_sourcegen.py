"""
Created on Aug 3, 2011

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
import ast
from ...asttools.visitors.pysourcegen import SourceGen
from ...asttools.tests import AllTypesTested
from ...testing import py2only, py3only
tested = AllTypesTested()

def simple_expr(expr):
    if False:
        print('Hello World!')

    def test_sourcegen_expr(self):
        if False:
            while True:
                i = 10
        self.assertSame(expr)
    return test_sourcegen_expr

def bin_op(op):
    if False:
        for i in range(10):
            print('nop')

    def test_bin_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertSame('(a %s b)' % (op,))
    return test_bin_op

def unary_op(op):
    if False:
        print('Hello World!')

    def test_bin_op(self):
        if False:
            while True:
                i = 10
        self.assertSame('(%sb)' % (op,))
    return test_bin_op

def aug_assign(op):
    if False:
        print('Hello World!')

    def test_bin_op(self):
        if False:
            i = 10
            return i + 15
        self.assertSame('a %s= b' % (op,))
    return test_bin_op

class Test(unittest.TestCase):

    def assertSame(self, source):
        if False:
            for i in range(10):
                print('nop')
        module = ast.parse(source)
        tested.update(module)
        gen = SourceGen()
        gen.visit(module)
        generated_source = gen.dumps()
        self.assertMultiLineEqual(source, generated_source.strip('\n'))

class TestSimple(Test):

    def assertSame(self, source):
        if False:
            return 10
        module = ast.parse(source)
        tested.update(module)
        gen = SourceGen()
        gen.visit(module)
        generated_source = gen.dumps()
        self.assertEqual(source, generated_source.strip('\n'))
    test_expr = simple_expr('a')
    test_del = simple_expr('del a')
    test_assign = simple_expr('a = 1')
    test_assign_multi = simple_expr('a = b = 1')
    test_attr = simple_expr('a.b')
    test_assattr = simple_expr('a.b = 1')
    test_index = simple_expr('a[b]')
    test_index2 = simple_expr('a[b, c]')
    test_slice0 = simple_expr('a[:]')
    test_slice1 = simple_expr('a[1:]')
    test_slice2 = simple_expr('a[1:2]')
    test_slice3 = simple_expr('a[1:2:3]')
    test_slice4 = simple_expr('a[1::3]')
    test_slice5 = simple_expr('a[::3]')
    test_slice6 = simple_expr('a[:3]')
    test_slice7 = simple_expr('a[...]')
    test_raise = simple_expr('raise Foo')
    test_raise1 = py2only(simple_expr('raise Foo, bar'))
    test_raise2 = py2only(simple_expr('raise Foo, bar, baz'))
    test_raise_from = py3only(simple_expr('raise Foo() from bar'))
    test_call0 = simple_expr('foo()')
    test_call1 = simple_expr('a = foo()')
    test_call2 = simple_expr('foo(x)')
    test_call3 = simple_expr('foo(x, y)')
    test_call4 = simple_expr('foo(x=y)')
    test_call5 = simple_expr('foo(z, x=y)')
    test_call6 = simple_expr('foo(*z)')
    test_call7 = simple_expr('foo(**z)')
    test_call8 = simple_expr('foo(a, b=c, *d, **z)')
    test_pass = simple_expr('pass')
    test_import = simple_expr('import a')
    test_import_as = simple_expr('import a as b')
    test_from_import = simple_expr('from c import a')
    test_from_import_as = simple_expr('from c import a as b')
    test_dict0 = simple_expr('{}')
    test_dict1 = simple_expr('{a:b}')
    test_dict2 = simple_expr('{a:b, c:d}')
    test_list0 = simple_expr('[]')
    test_list1 = simple_expr('[a]')
    test_list2 = simple_expr('[a, b]')
    test_set1 = simple_expr('{a}')
    test_set2 = simple_expr('{a, b}')
    test_exec0 = py2only(simple_expr('exec a in None, None'))
    test_exec1 = py2only(simple_expr('exec a in b, None'))
    test_exec2 = py2only(simple_expr('exec a in b, c'))
    test_assert1 = simple_expr('assert False')
    test_assert2 = simple_expr('assert False, msg')
    test_global1 = simple_expr('global a')
    test_global2 = simple_expr('global a, b')
    test_str = simple_expr("x = 'a'")
    test_ifexpr = simple_expr('a = b if c else d')
    test_lambda = simple_expr('a = lambda a: a')
    test_list_comp = simple_expr('[a for b in c]')
    test_list_comp_if = simple_expr('[a for b in c if d]')
    test_list_comp_if2 = simple_expr('[a for b in c if d if e]')
    test_list_comp2 = simple_expr('[a for b in c for d in e]')
    test_list_comp3 = simple_expr('[a for b in c for d in e if k for f in g]')
    test_set_comp = simple_expr('{a for b in c}')
    test_dict_comp = simple_expr('{a:d for b in c}')
    test_iadd = aug_assign('+')
    test_isub = aug_assign('-')
    test_imult = aug_assign('*')
    test_ipow = aug_assign('**')
    test_idiv = aug_assign('/')
    test_ifdiv = aug_assign('//')
    test_add = bin_op('+')
    test_sub = bin_op('-')
    test_mult = bin_op('*')
    test_pow = bin_op('**')
    test_div = bin_op('/')
    test_floordiv = bin_op('//')
    test_mod = bin_op('%')
    test_eq = bin_op('==')
    test_neq = bin_op('!=')
    test_lt = bin_op('<')
    test_gt = bin_op('>')
    test_lte = bin_op('<=')
    test_gte = bin_op('>=')
    test_lshift = bin_op('<<')
    test_rshift = bin_op('>>')
    test_lshift = bin_op('and')
    test_rshift = bin_op('or')
    test_in = bin_op('in')
    test_not_in = bin_op('not in')
    test_is = bin_op('is')
    test_is_not = bin_op('is not')
    test_bitand = bin_op('&')
    test_bitor = bin_op('|')
    test_bitxor = bin_op('^')
    test_usub = unary_op('-')
    test_uadd = unary_op('+')
    test_unot = unary_op('not ')
    test_uinvert = unary_op('~')

class ControlFlow(Test):

    def test_if(self):
        if False:
            while True:
                i = 10
        source = 'if a:\n    b'
        self.assertSame(source)

    def test_if_else(self):
        if False:
            while True:
                i = 10
        source = 'if a:\n    b\nelse:\n    c'
        self.assertSame(source)

    def test_elif_else(self):
        if False:
            while True:
                i = 10
        source = 'if a:\n    b\nelif d:\n    e\nelse:\n    c'
        self.assertSame(source)

    def test_while(self):
        if False:
            while True:
                i = 10
        source = 'while a:\n    b'
        self.assertSame(source)

    def test_break(self):
        if False:
            while True:
                i = 10
        source = 'while a:\n    break'
        self.assertSame(source)

    def test_continue(self):
        if False:
            print('Hello World!')
        source = 'while a:\n    continue'
        self.assertSame(source)

    def test_with0(self):
        if False:
            print('Hello World!')
        source = 'with a:\n    b'
        self.assertSame(source)

    def test_with1(self):
        if False:
            i = 10
            return i + 15
        source = 'with a as b:\n    c'
        self.assertSame(source)

    def test_function_def(self):
        if False:
            return 10
        source = 'def foo():\n    pass'
        self.assertSame(source)

    def test_return(self):
        if False:
            while True:
                i = 10
        source = 'def foo():\n    return 1.1'
        self.assertSame(source)

    def test_yield(self):
        if False:
            return 10
        source = 'def foo():\n    yield 1.1'
        self.assertSame(source)

    def test_function_args1(self):
        if False:
            return 10
        source = 'def foo(a):\n    pass'
        self.assertSame(source)

    def test_function_args2(self):
        if False:
            return 10
        source = 'def foo(a, b):\n    pass'
        self.assertSame(source)

    def test_function_args3(self):
        if False:
            return 10
        source = 'def foo(b=c):\n    pass'
        self.assertSame(source)

    def test_function_args4(self):
        if False:
            for i in range(10):
                print('nop')
        source = 'def foo(b=c, d=e):\n    pass'
        self.assertSame(source)

    def test_function_args5(self):
        if False:
            for i in range(10):
                print('nop')
        source = 'def foo(*a):\n    pass'
        self.assertSame(source)

    def test_try_except(self):
        if False:
            i = 10
            return i + 15
        source = 'try:\n    a\nexcept:\n    b'
        self.assertSame(source)

    def test_try_except1(self):
        if False:
            print('Hello World!')
        source = 'try:\n    a\nexcept Exception:\n    b'
        self.assertSame(source)

    def test_try_except2(self):
        if False:
            i = 10
            return i + 15
        source = 'try:\n    a\nexcept Exception as error:\n    b'
        self.assertSame(source)

    def test_try_except3(self):
        if False:
            return 10
        source = 'try:\n    a\nexcept Exception as error:\n    pass\nexcept:\n    b'
        self.assertSame(source)

    def test_try_except_else(self):
        if False:
            return 10
        source = 'try:\n    a\nexcept Exception as error:\n    pass\nexcept:\n    b\nelse:\n    c'
        self.assertSame(source)

    def test_try_except_finally(self):
        if False:
            for i in range(10):
                print('nop')
        source = 'try:\n    a\nexcept Exception as error:\n    pass\nexcept:\n    b\nfinally:\n    c'
        self.assertSame(source)

    def test_for(self):
        if False:
            for i in range(10):
                print('nop')
        source = 'for i in j:\n    pass'
        self.assertSame(source)

    def test_for_else(self):
        if False:
            for i in range(10):
                print('nop')
        source = 'for i in j:\n    l\nelse:\n    k'
        self.assertSame(source)

    def test_class_def(self):
        if False:
            return 10
        source = 'class A():\n    pass'
        self.assertSame(source)

    def test_class_def1(self):
        if False:
            while True:
                i = 10
        source = 'class A(object):\n    pass'
        self.assertSame(source)

    def test_class_def2(self):
        if False:
            i = 10
            return i + 15
        source = 'class A(object, foo):\n    pass'
        self.assertSame(source)

    def test_class_def3(self):
        if False:
            print('Hello World!')
        source = 'class A(object, foo):\n    a = 1\n    def bar():\n        pass'
        self.assertSame(source)
if __name__ == '__main__':
    unittest.main(exit=False)
    print(tested.tested())