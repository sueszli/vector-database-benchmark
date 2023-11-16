"""Module with basic entity definitions for testing."""

def simple_function(x):
    if False:
        print('Hello World!')
    'Docstring.'
    return x

def nested_functions(x):
    if False:
        return 10
    'Docstring.'

    def inner_fn(y):
        if False:
            for i in range(10):
                print('nop')
        return y
    return inner_fn(x)

def function_with_print():
    if False:
        for i in range(10):
            print('nop')
    print('foo')
simple_lambda = lambda : None

class SimpleClass(object):

    def simple_method(self):
        if False:
            print('Hello World!')
        return self

    def method_with_print(self):
        if False:
            print('Hello World!')
        print('foo')

def function_with_multiline_call(x):
    if False:
        i = 10
        return i + 15
    'Docstring.'
    return range(x, x + 1)

def basic_decorator(f):
    if False:
        print('Hello World!')
    return f

@basic_decorator
@basic_decorator
def decorated_function(x):
    if False:
        for i in range(10):
            print('nop')
    if x > 0:
        return 1
    return 2