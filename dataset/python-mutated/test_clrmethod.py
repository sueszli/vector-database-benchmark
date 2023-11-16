"""Test clrmethod and clrproperty support for calling methods and getting/setting python properties from CLR."""
import Python.Test as Test
import System
import pytest
import clr

class ExampleClrClass(System.Object):
    __namespace__ = 'PyTest'

    def __init__(self):
        if False:
            return 10
        self._x = 3

    @clr.clrmethod(int, [int])
    def test(self, x):
        if False:
            i = 10
            return i + 15
        return x * 2

    def get_X(self):
        if False:
            return 10
        return self._x

    def set_X(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._x = value
    X = clr.clrproperty(int, get_X, set_X)

    @clr.clrproperty(int)
    def Y(self):
        if False:
            print('Hello World!')
        return self._x * 2

def test_set_and_get_property_from_py():
    if False:
        while True:
            i = 10
    'Test setting and getting clr-accessible properties from python.'
    t = ExampleClrClass()
    assert t.X == 3
    assert t.Y == 3 * 2
    t.X = 4
    assert t.X == 4
    assert t.Y == 4 * 2

def test_set_and_get_property_from_clr():
    if False:
        i = 10
        return i + 15
    'Test setting and getting clr-accessible properties from the clr.'
    t = ExampleClrClass()
    assert t.GetType().GetProperty('X').GetValue(t) == 3
    assert t.GetType().GetProperty('Y').GetValue(t) == 3 * 2
    t.GetType().GetProperty('X').SetValue(t, 4)
    assert t.GetType().GetProperty('X').GetValue(t) == 4
    assert t.GetType().GetProperty('Y').GetValue(t) == 4 * 2

def test_set_and_get_property_from_clr_and_py():
    if False:
        return 10
    'Test setting and getting clr-accessible properties alternatingly from the clr and from python.'
    t = ExampleClrClass()
    assert t.GetType().GetProperty('X').GetValue(t) == 3
    assert t.GetType().GetProperty('Y').GetValue(t) == 3 * 2
    assert t.X == 3
    assert t.Y == 3 * 2
    t.GetType().GetProperty('X').SetValue(t, 4)
    assert t.GetType().GetProperty('X').GetValue(t) == 4
    assert t.GetType().GetProperty('Y').GetValue(t) == 4 * 2
    assert t.X == 4
    assert t.Y == 4 * 2
    t.X = 5
    assert t.GetType().GetProperty('X').GetValue(t) == 5
    assert t.GetType().GetProperty('Y').GetValue(t) == 5 * 2
    assert t.X == 5
    assert t.Y == 5 * 2

def test_method_invocation_from_py():
    if False:
        return 10
    'Test calling a clr-accessible method from python.'
    t = ExampleClrClass()
    assert t.test(41) == 41 * 2

def test_method_invocation_from_clr():
    if False:
        while True:
            i = 10
    'Test calling a clr-accessible method from the clr.'
    t = ExampleClrClass()
    assert t.GetType().GetMethod('test').Invoke(t, [37]) == 37 * 2