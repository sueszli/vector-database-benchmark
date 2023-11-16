"""
Created on Jul 14, 2011

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
from ...testing import py2, py2only
from ...decompiler.tests import Base
filename = 'tests.py'

class LogicJumps(Base):

    def test_logic1(self):
        if False:
            return 10
        'a and b or c'
        self.statement('a and b or c')

    def test_logic2(self):
        if False:
            for i in range(10):
                print('nop')
        'a or (b or c)'
        self.statement('a or (b or c)')

    def test_if_expr_discard(self):
        if False:
            while True:
                i = 10
        stmnt = 'a if b else c'
        self.statement(stmnt)

    @unittest.skip('I think this may be a bug in python')
    def test_if_expr_const_bug(self):
        if False:
            i = 10
            return i + 15
        stmnt = '0 if 1 else 2'
        self.statement(stmnt)

    def test_if_expr_assign(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = 'd = a if b else c'
        self.statement(stmnt)

    def test_if_expr_assignattr(self):
        if False:
            i = 10
            return i + 15
        stmnt = 'd.a = a if b else c'
        self.statement(stmnt)

    def test_bug010(self):
        if False:
            while True:
                i = 10
        stmnt = '\ndef foo():\n    if a:\n        return 1\n    else:\n        return 2\n        '
        equiv = '\ndef foo():\n    if a:\n        return 1\n    return 2\n    return None\n        '
        self.statement(stmnt, equiv=equiv)

    @unittest.expectedFailure
    def test_bug011(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\ndef foo():\n    if a or b or c:\n        return 1\n    else:\n        return 2\n        '
        self.statement(stmnt)

class Function(Base):

    def test_function(self):
        if False:
            print('Hello World!')
        stmnt = '\ndef foo():\n    return None\n'
        self.statement(stmnt)

    def test_function_args(self):
        if False:
            i = 10
            return i + 15
        stmnt = "\ndef foo(a, b, c='asdf'):\n    return None\n"
        self.statement(stmnt)

    def test_function_var_args(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\ndef foo(a, b, *c):\n    return None\n'
        self.statement(stmnt)

    def test_function_varkw_args(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\ndef foo(a, b, *c, **d):\n    return None\n'
        self.statement(stmnt)

    def test_function_kw_args(self):
        if False:
            i = 10
            return i + 15
        stmnt = '\ndef foo(a, b, **d):\n    return None\n'
        self.statement(stmnt)

    def test_function_yield(self):
        if False:
            i = 10
            return i + 15
        stmnt = '\ndef foo(a, b):\n    yield a + b\n    return\n'
        self.statement(stmnt)

    def test_function_decorator(self):
        if False:
            print('Hello World!')
        stmnt = '\n@bar\ndef foo(a, b):\n    return None\n'
        self.statement(stmnt)

    def test_function_decorator2(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\n@bar\n@bar2\ndef foo(a, b):\n    return None\n'
        self.statement(stmnt)

    def test_build_lambda(self):
        if False:
            return 10
        stmnt = 'lambda a: a'
        self.statement(stmnt)

    def test_build_lambda1(self):
        if False:
            while True:
                i = 10
        stmnt = 'func = lambda a, b: a+1'
        self.statement(stmnt)

    def test_build_lambda_var_args(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = 'func = lambda a, *b: a+1'
        self.statement(stmnt)

    def test_build_lambda_kw_args(self):
        if False:
            print('Hello World!')
        stmnt = 'func = lambda **b: a+1'
        self.statement(stmnt)

    def test_build_lambda_varkw_args(self):
        if False:
            while True:
                i = 10
        stmnt = 'func = lambda *a, **b: a+1'
        self.statement(stmnt)

class ClassDef(Base):

    def test_build_class(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = "\nclass Bar(object):\n    'adsf'\n    a = 1\n"
        self.statement(stmnt)

    def test_build_class_wfunc(self):
        if False:
            return 10
        stmnt = "\nclass Bar(object):\n    'adsf'\n    a = 1\n    def foo(self):\n        return None\n\n"
        self.statement(stmnt)

    def test_build_class_wdec(self):
        if False:
            while True:
                i = 10
        stmnt = "\n@decorator\nclass Bar(object):\n    'adsf'\n    a = 1\n    def foo(self):\n        return None\n\n"
        self.statement(stmnt)

class ControlFlow(Base):

    def test_if(self):
        if False:
            print('Hello World!')
        self.statement('if a: b')

    def test_if2(self):
        if False:
            print('Hello World!')
        self.statement('if a: b or c')

    def test_if3(self):
        if False:
            return 10
        self.statement('if a and b: c')

    def test_if4(self):
        if False:
            return 10
        self.statement('if a or b: c')

    def test_if5(self):
        if False:
            print('Hello World!')
        self.statement('if not a: c')

    def test_if6(self):
        if False:
            i = 10
            return i + 15
        self.statement('if not a or b: c')

    def test_elif(self):
        if False:
            print('Hello World!')
        stmnt = 'if a:\n    b\nelif c:\n    d'
        self.statement(stmnt)

    def test_if_else(self):
        if False:
            print('Hello World!')
        stmnt = 'if a:\n    b\nelse:\n    d'
        self.statement(stmnt)

    def test_if_elif_else(self):
        if False:
            return 10
        stmnt = 'if a:\n    b\nelif f:\n    d\nelse:\n    d'
        self.statement(stmnt)

    def test_tryexcept1(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\ntry:\n    foo\nexcept:\n    bar\n'
        self.statement(stmnt)

    def test_tryexcept_else(self):
        if False:
            while True:
                i = 10
        stmnt = '\ntry:\n    foo\nexcept:\n    bar\nelse:\n    baz\n'
        self.statement(stmnt)

    def test_tryexcept2(self):
        if False:
            print('Hello World!')
        stmnt = '\ntry:\n    foo\nexcept Exception:\n    bar\nelse:\n    baz\n'
        self.statement(stmnt)

    def test_tryexcept3(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\ntry:\n    foo\nexcept Exception as error:\n    bar\nelse:\n    baz\n'
        self.statement(stmnt)

    def test_tryexcept4(self):
        if False:
            i = 10
            return i + 15
        stmnt = '\ntry:\n    foo\nexcept Exception as error:\n    bar\nexcept Baz as error:\n    bar\nelse:\n    baz\n'
        self.statement(stmnt)

    def test_while(self):
        if False:
            i = 10
            return i + 15
        self.statement('while b: a')

    def test_while1(self):
        if False:
            i = 10
            return i + 15
        self.statement('while 1: a')

    def test_while_logic(self):
        if False:
            while True:
                i = 10
        self.statement('while a or b: x')

    def test_while_logic2(self):
        if False:
            while True:
                i = 10
        self.statement('while a and b: x')

    def test_while_logic3(self):
        if False:
            return 10
        self.statement('while a >= r and b == c: x')

    def test_while_else(self):
        if False:
            while True:
                i = 10
        stmnt = '\nwhile a:\n    break\nelse:\n    a\n'
        self.statement(stmnt)

    def test_for(self):
        if False:
            print('Hello World!')
        stmnt = '\nfor i in  a:\n    break\n'
        self.statement(stmnt)

    def test_for2(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\nfor i in  a:\n    b = 3\n'
        self.statement(stmnt)

    def test_for_else(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\nfor i in  a:\n    b = 3\nelse:\n    b= 2\n'
        self.statement(stmnt)

    def test_for_continue(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\nfor i in  a:\n    b = 3\n    continue\n'
        self.statement(stmnt)

    def test_for_unpack(self):
        if False:
            while True:
                i = 10
        stmnt = '\nfor i,j in  a:\n    b = 3\n'
        self.statement(stmnt)

    def test_try_continue(self):
        if False:
            print('Hello World!')
        stmnt = '\nfor x in (1,2):\n        try: continue\n        except: pass\n'
        self.statement(stmnt)

    def test_loop_01(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\nif c > d:\n    if e > f:\n        g\n    h\n'

    def test_loop_bug(self):
        if False:
            print('Hello World!')
        stmnt = '\nfor a in b:\n    if c > d:\n        if e > f:\n            g\n        h\n'
        self.statement(stmnt)

    def test_while_bug(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\nwhile a:\n    q\n    while b:\n        w\n'
        self.statement(stmnt)

    @unittest.expectedFailure
    def test_while_bug02(self):
        if False:
            while True:
                i = 10
        stmnt = '\nwhile 1:\n    b += y\n    if b < x:\n        break\n'
        self.statement(stmnt)

class Complex(Base):

    def test_if_in_for(self):
        if False:
            while True:
                i = 10
        stmnt = '\nfor i in j:\n    if i:\n        j =1\n'
        self.statement(stmnt)

    def test_if_in_for2(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\nfor i in j:\n    if i:\n        a\n    else:\n        b\n\n'
        self.statement(stmnt)

    def test_if_in_for3(self):
        if False:
            i = 10
            return i + 15
        stmnt = '\nfor i in j:\n    if i:\n        break\n    else:\n        continue\n\n'
        equiv = '\nfor i in j:\n    if i:\n        break\n        continue\n\n'
        self.statement(stmnt, equiv)

    def test_if_in_while(self):
        if False:
            while True:
                i = 10
        stmnt = '\nwhile i in j:\n    if i:\n        a\n    else:\n        b\n\n'
        self.statement(stmnt)

    def test_nested_if(self):
        if False:
            print('Hello World!')
        stmnt = '\nif a:\n    if b:\n        c\n    else:\n        d\n'
        self.statement(stmnt)

    def test_nested_if2(self):
        if False:
            while True:
                i = 10
        stmnt = '\nif a:\n    if b:\n        c\n    else:\n        d\nelse:\n    b\n'
        self.statement(stmnt)

    def test_if_return(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\ndef a():\n    if b:\n        return None\n    return None\n'
        self.statement(stmnt)

    def test_if_return2(self):
        if False:
            for i in range(10):
                print('nop')
        stmnt = '\ndef a():\n    if b:\n        a\n    else:\n        return b\n\n    return c\n'
        self.statement(stmnt)

    def test_nested_while_bug(self):
        if False:
            i = 10
            return i + 15
        stmnt = '\nif gid == 0:\n    output[0] = initial\n    while i < input.size:\n        output[0] += shared[i]\n'
        self.statement(stmnt)

    def test_aug_assign_slice(self):
        if False:
            print('Hello World!')
        stmnt = 'c[idx:a:3] += b[idx:a]'
        self.statement(stmnt)

    def test_issue_4(self):
        if False:
            return 10
        example = '\ndef example(idx):\n   if(idx == 2 or idx == 3):\n      idx = 1\n      return None\n   i += 1\n   return None\n        '
        self.statement(example)
if __name__ == '__main__':
    unittest.main()