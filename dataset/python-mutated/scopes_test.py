"""Tests slim.scopes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from inception.slim import scopes

@scopes.add_arg_scope
def func1(*args, **kwargs):
    if False:
        return 10
    return (args, kwargs)

@scopes.add_arg_scope
def func2(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    return (args, kwargs)

class ArgScopeTest(tf.test.TestCase):

    def testEmptyArgScope(self):
        if False:
            print('Hello World!')
        with self.test_session():
            self.assertEqual(scopes._current_arg_scope(), {})

    def testCurrentArgScope(self):
        if False:
            i = 10
            return i + 15
        func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
        key_op = (func1.__module__, func1.__name__)
        current_scope = {key_op: func1_kwargs.copy()}
        with self.test_session():
            with scopes.arg_scope([func1], a=1, b=None, c=[1]) as scope:
                self.assertDictEqual(scope, current_scope)

    def testCurrentArgScopeNested(self):
        if False:
            while True:
                i = 10
        func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
        func2_kwargs = {'b': 2, 'd': [2]}
        key = lambda f: (f.__module__, f.__name__)
        current_scope = {key(func1): func1_kwargs.copy(), key(func2): func2_kwargs.copy()}
        with self.test_session():
            with scopes.arg_scope([func1], a=1, b=None, c=[1]):
                with scopes.arg_scope([func2], b=2, d=[2]) as scope:
                    self.assertDictEqual(scope, current_scope)

    def testReuseArgScope(self):
        if False:
            while True:
                i = 10
        func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
        key_op = (func1.__module__, func1.__name__)
        current_scope = {key_op: func1_kwargs.copy()}
        with self.test_session():
            with scopes.arg_scope([func1], a=1, b=None, c=[1]) as scope1:
                pass
            with scopes.arg_scope(scope1) as scope:
                self.assertDictEqual(scope, current_scope)

    def testReuseArgScopeNested(self):
        if False:
            return 10
        func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
        func2_kwargs = {'b': 2, 'd': [2]}
        key = lambda f: (f.__module__, f.__name__)
        current_scope1 = {key(func1): func1_kwargs.copy()}
        current_scope2 = {key(func1): func1_kwargs.copy(), key(func2): func2_kwargs.copy()}
        with self.test_session():
            with scopes.arg_scope([func1], a=1, b=None, c=[1]) as scope1:
                with scopes.arg_scope([func2], b=2, d=[2]) as scope2:
                    pass
            with scopes.arg_scope(scope1):
                self.assertDictEqual(scopes._current_arg_scope(), current_scope1)
            with scopes.arg_scope(scope2):
                self.assertDictEqual(scopes._current_arg_scope(), current_scope2)

    def testSimpleArgScope(self):
        if False:
            for i in range(10):
                print('nop')
        func1_args = (0,)
        func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
        with self.test_session():
            with scopes.arg_scope([func1], a=1, b=None, c=[1]):
                (args, kwargs) = func1(0)
                self.assertTupleEqual(args, func1_args)
                self.assertDictEqual(kwargs, func1_kwargs)

    def testSimpleArgScopeWithTuple(self):
        if False:
            print('Hello World!')
        func1_args = (0,)
        func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
        with self.test_session():
            with scopes.arg_scope((func1,), a=1, b=None, c=[1]):
                (args, kwargs) = func1(0)
                self.assertTupleEqual(args, func1_args)
                self.assertDictEqual(kwargs, func1_kwargs)

    def testOverwriteArgScope(self):
        if False:
            i = 10
            return i + 15
        func1_args = (0,)
        func1_kwargs = {'a': 1, 'b': 2, 'c': [1]}
        with scopes.arg_scope([func1], a=1, b=None, c=[1]):
            (args, kwargs) = func1(0, b=2)
            self.assertTupleEqual(args, func1_args)
            self.assertDictEqual(kwargs, func1_kwargs)

    def testNestedArgScope(self):
        if False:
            return 10
        func1_args = (0,)
        func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
        with scopes.arg_scope([func1], a=1, b=None, c=[1]):
            (args, kwargs) = func1(0)
            self.assertTupleEqual(args, func1_args)
            self.assertDictEqual(kwargs, func1_kwargs)
            func1_kwargs['b'] = 2
            with scopes.arg_scope([func1], b=2):
                (args, kwargs) = func1(0)
                self.assertTupleEqual(args, func1_args)
                self.assertDictEqual(kwargs, func1_kwargs)

    def testSharedArgScope(self):
        if False:
            i = 10
            return i + 15
        func1_args = (0,)
        func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
        with scopes.arg_scope([func1, func2], a=1, b=None, c=[1]):
            (args, kwargs) = func1(0)
            self.assertTupleEqual(args, func1_args)
            self.assertDictEqual(kwargs, func1_kwargs)
            (args, kwargs) = func2(0)
            self.assertTupleEqual(args, func1_args)
            self.assertDictEqual(kwargs, func1_kwargs)

    def testSharedArgScopeTuple(self):
        if False:
            i = 10
            return i + 15
        func1_args = (0,)
        func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
        with scopes.arg_scope((func1, func2), a=1, b=None, c=[1]):
            (args, kwargs) = func1(0)
            self.assertTupleEqual(args, func1_args)
            self.assertDictEqual(kwargs, func1_kwargs)
            (args, kwargs) = func2(0)
            self.assertTupleEqual(args, func1_args)
            self.assertDictEqual(kwargs, func1_kwargs)

    def testPartiallySharedArgScope(self):
        if False:
            for i in range(10):
                print('nop')
        func1_args = (0,)
        func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
        func2_args = (1,)
        func2_kwargs = {'a': 1, 'b': None, 'd': [2]}
        with scopes.arg_scope([func1, func2], a=1, b=None):
            with scopes.arg_scope([func1], c=[1]), scopes.arg_scope([func2], d=[2]):
                (args, kwargs) = func1(0)
                self.assertTupleEqual(args, func1_args)
                self.assertDictEqual(kwargs, func1_kwargs)
                (args, kwargs) = func2(1)
                self.assertTupleEqual(args, func2_args)
                self.assertDictEqual(kwargs, func2_kwargs)
if __name__ == '__main__':
    tf.test.main()