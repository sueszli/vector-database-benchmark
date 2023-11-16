""" Test that covers attributes if function objects.

"""
from __future__ import print_function

def func(_arg1, _arg2, _arg3, **_star):
    if False:
        print('Hello World!')
    'Some documentation.'
print('Starting out: func, __name__:', func, func.__name__)
print('Changing its name:')
func.__name__ = 'renamed'
print('With new name: func, __name__:', func, func.__name__)
print('Documentation initially:', func.__doc__)
print('Changing its doc:')
func.__doc__ = 'changed doc' + chr(0) + ' with 0 character'
print('Documentation updated:', repr(func.__doc__))
print('Setting its dict')
func.my_value = 'attached value'
print('Reading its dict', func.my_value)
print('__code__ dir')
print(dir(func.__code__))

def func2(_arg1, arg2='default_arg2', arg3='default_arg3'):
    if False:
        while True:
            i = 10
    x = arg2 + arg3
    return x
print('func __defaults__', func2.__defaults__)
print('function varnames', func2.__code__.co_varnames)