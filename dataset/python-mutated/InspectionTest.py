"""Tests uncompiled functions and compiled functions responses to inspect and isistance.  """
from __future__ import print_function
import inspect
import pprint
import sys
import types

def displayDict(d, remove_keys=()):
    if False:
        while True:
            i = 10
    if '__loader__' in d:
        d = dict(d)
        if str is bytes:
            del d['__loader__']
        else:
            d['__loader__'] = '<__loader__ removed>'
    if '__file__' in d:
        d = dict(d)
        d['__file__'] = '<__file__ removed>'
    if '__compiled__' in d:
        d = dict(d)
        del d['__compiled__']
    for remove_key in remove_keys:
        if remove_key in d:
            d = dict(d)
            del d[remove_key]
    return pprint.pformat(d)

def compiledFunction(a, b):
    if False:
        for i in range(10):
            print('nop')
    pass
print('Function inspect.isfunction:', inspect.isfunction(compiledFunction))
print('Function isinstance types.FunctionType:', isinstance(compiledFunction, types.FunctionType))
print('Function isinstance tuple containing types.FunctionType:', isinstance(compiledFunction, (int, types.FunctionType)))
assert type(compiledFunction) == types.FunctionType

class CompiledClass:

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def compiledMethod(self):
        if False:
            while True:
                i = 10
        pass
assert inspect.isfunction(CompiledClass) is False
assert isinstance(CompiledClass, types.FunctionType) is False
assert inspect.ismethod(compiledFunction) is False
assert inspect.ismethod(CompiledClass) is False
assert inspect.ismethod(CompiledClass.compiledMethod) == (sys.version_info < (3,))
assert inspect.ismethod(CompiledClass().compiledMethod) is True
assert bool(type(CompiledClass.compiledMethod) == types.MethodType) == (sys.version_info < (3,))

def compiledGenerator():
    if False:
        i = 10
        return i + 15
    yield 1
assert inspect.isfunction(compiledGenerator) is True
assert inspect.isgeneratorfunction(compiledGenerator) is True
assert isinstance(compiledGenerator(), types.GeneratorType) is True
assert type(compiledGenerator()) == types.GeneratorType
assert isinstance(compiledGenerator, types.GeneratorType) is False
assert inspect.ismethod(compiledGenerator()) is False
assert inspect.isfunction(compiledGenerator()) is False
assert inspect.isgenerator(compiledFunction) is False
assert inspect.isgenerator(compiledGenerator) is False
assert inspect.isgenerator(compiledGenerator()) is True

def someFunction(a):
    if False:
        while True:
            i = 10
    assert inspect.isframe(sys._getframe())
someFunction(2)

class C:
    print('Class locals', displayDict(sys._getframe().f_locals, remove_keys=('__qualname__', '__locals__')))
    print('Class flags', sys._getframe().f_code.co_flags)

def f():
    if False:
        i = 10
        return i + 15
    print('Func locals', sys._getframe().f_locals)
    print('Func flags', sys._getframe().f_code.co_flags)
f()

def g():
    if False:
        for i in range(10):
            print('nop')
    yield ('Generator object locals', sys._getframe().f_locals)
    yield ('Generator object flags', sys._getframe().f_code.co_flags)
for line in g():
    print(*line)
print('Generator function flags', g.__code__.co_flags)
print('Module frame locals', displayDict(sys._getframe().f_locals))
print('Module flags', sys._getframe().f_code.co_flags)
print('Module code name', sys._getframe().f_code.co_name)
print('Module frame dir', dir(sys._getframe()))