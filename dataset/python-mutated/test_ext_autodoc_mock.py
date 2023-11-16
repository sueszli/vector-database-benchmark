"""Test the autodoc extension."""
from __future__ import annotations
import abc
import sys
from importlib import import_module
from typing import TypeVar
import pytest
from sphinx.ext.autodoc.mock import _MockModule, _MockObject, ismock, mock, undecorate

def test_MockModule():
    if False:
        print('Hello World!')
    mock = _MockModule('mocked_module')
    assert isinstance(mock.some_attr, _MockObject)
    assert isinstance(mock.some_method, _MockObject)
    assert isinstance(mock.attr1.attr2, _MockObject)
    assert isinstance(mock.attr1.attr2.meth(), _MockObject)
    assert repr(mock.some_attr) == 'mocked_module.some_attr'
    assert repr(mock.some_method) == 'mocked_module.some_method'
    assert repr(mock.attr1.attr2) == 'mocked_module.attr1.attr2'
    assert repr(mock.attr1.attr2.meth) == 'mocked_module.attr1.attr2.meth'
    assert repr(mock) == 'mocked_module'

def test_MockObject():
    if False:
        return 10
    mock = _MockObject()
    assert isinstance(mock.some_attr, _MockObject)
    assert isinstance(mock.some_method, _MockObject)
    assert isinstance(mock.attr1.attr2, _MockObject)
    assert isinstance(mock.attr1.attr2.meth(), _MockObject)

    class SubClass(mock.SomeClass):
        """docstring of SubClass"""

        def method(self):
            if False:
                while True:
                    i = 10
            return 'string'
    obj = SubClass()
    assert SubClass.__doc__ == 'docstring of SubClass'
    assert isinstance(obj, SubClass)
    assert obj.method() == 'string'
    assert isinstance(obj.other_method(), SubClass)
    T = TypeVar('T')

    class SubClass2(mock.SomeClass[T]):
        """docstring of SubClass"""
    obj2 = SubClass2()
    assert SubClass2.__doc__ == 'docstring of SubClass'
    assert isinstance(obj2, SubClass2)

def test_mock():
    if False:
        while True:
            i = 10
    modname = 'sphinx.unknown'
    submodule = modname + '.submodule'
    assert modname not in sys.modules
    with pytest.raises(ImportError):
        import_module(modname)
    with mock([modname]):
        import_module(modname)
        assert modname in sys.modules
        assert isinstance(sys.modules[modname], _MockModule)
        import_module(submodule)
        assert submodule in sys.modules
        assert isinstance(sys.modules[submodule], _MockModule)
    assert modname not in sys.modules
    with pytest.raises(ImportError):
        import_module(modname)

def test_mock_does_not_follow_upper_modules():
    if False:
        return 10
    with mock(['sphinx.unknown.module']):
        with pytest.raises(ImportError):
            import_module('sphinx.unknown')

def test_abc_MockObject():
    if False:
        return 10
    mock = _MockObject()

    class Base:

        @abc.abstractmethod
        def __init__(self):
            if False:
                i = 10
                return i + 15
            pass

    class Derived(Base, mock.SubClass):
        pass
    obj = Derived()
    assert isinstance(obj, Base)
    assert isinstance(obj, _MockObject)
    assert isinstance(obj.some_method(), Derived)

def test_mock_decorator():
    if False:
        i = 10
        return i + 15
    mock = _MockObject()

    @mock.function_deco
    def func():
        if False:
            while True:
                i = 10
        pass

    class Foo:

        @mock.method_deco
        def meth(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        @classmethod
        @mock.method_deco
        def class_meth(cls):
            if False:
                return 10
            pass

    @mock.class_deco
    class Bar:
        pass

    @mock.funcion_deco(Foo)
    class Baz:
        pass
    assert undecorate(func).__name__ == 'func'
    assert undecorate(Foo.meth).__name__ == 'meth'
    assert undecorate(Foo.class_meth).__name__ == 'class_meth'
    assert undecorate(Bar).__name__ == 'Bar'
    assert undecorate(Baz).__name__ == 'Baz'

def test_ismock():
    if False:
        return 10
    with mock(['sphinx.unknown']):
        mod1 = import_module('sphinx.unknown')
        mod2 = import_module('sphinx.application')

        class Inherited(mod1.Class):
            pass
        assert ismock(mod1) is True
        assert ismock(mod1.Class) is True
        assert ismock(mod1.submod.Class) is True
        assert ismock(Inherited) is False
        assert ismock(mod2) is False
        assert ismock(mod2.Sphinx) is False