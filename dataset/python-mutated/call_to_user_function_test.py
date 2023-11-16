"""Calls to dynamic (i.e. nonglobal) functions.

Examples:
 * function variables
 * function parameters
 * factories
"""
import tensorflow as tf
from tensorflow.python.autograph.tests import reference_test_base

def function_1(x):
    if False:
        while True:
            i = 10
    return x * x * x

def function_2(x):
    if False:
        print('Hello World!')
    return -1 * x + 11

def factory(n):
    if False:
        for i in range(10):
            print('nop')
    if n == 1:
        return function_1
    return function_2

def static_fn(x):
    if False:
        for i in range(10):
            print('nop')
    a = function_1(x)
    b = function_2(x)
    return a + b

def factory_dynamic_fn(x):
    if False:
        print('Hello World!')
    f = factory(1)
    a = f(x)
    f = factory(2)
    b = f(x)
    return a + b

def param_dynamic_fn(f, x):
    if False:
        print('Hello World!')
    return f(x)

def variable_dynamic_fn(x):
    if False:
        while True:
            i = 10
    f = function_1
    a = f(x)
    f = function_2
    b = f(x)
    return a + b

def variable_dynamic_whitelisted_fn(x):
    if False:
        i = 10
        return i + 15
    f = tf.identity
    return f(x)

def dynamic_fn_with_kwargs(f, x):
    if False:
        while True:
            i = 10
    return f(x=x)

class ReferenceTest(reference_test_base.TestCase):

    def test_basic(self):
        if False:
            return 10
        self.assertFunctionMatchesEager(static_fn, 1)
        self.assertFunctionMatchesEager(factory_dynamic_fn, 1)
        self.assertFunctionMatchesEager(param_dynamic_fn, function_1, 1)
        self.assertFunctionMatchesEager(variable_dynamic_fn, 1)
        self.assertFunctionMatchesEager(variable_dynamic_whitelisted_fn, 1)
        self.assertFunctionMatchesEager(dynamic_fn_with_kwargs, function_1, 1)

    def test_basic_tensor(self):
        if False:
            print('Hello World!')
        self.all_inputs_tensors = True
        self.assertFunctionMatchesEager(static_fn, 1)
        self.assertFunctionMatchesEager(factory_dynamic_fn, 1)
        self.assertFunctionMatchesEager(param_dynamic_fn, function_1, 1)
        self.assertFunctionMatchesEager(variable_dynamic_fn, 1)
        self.assertFunctionMatchesEager(variable_dynamic_whitelisted_fn, 1)
        self.assertFunctionMatchesEager(dynamic_fn_with_kwargs, function_1, 1)
if __name__ == '__main__':
    tf.test.main()