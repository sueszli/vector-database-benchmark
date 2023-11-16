"""Tests for :mod:`sphinx.ext.napoleon.__init__` module."""
import functools
from collections import namedtuple
from unittest import mock
import pytest
from sphinx.application import Sphinx
from sphinx.ext.napoleon import Config, _process_docstring, _skip_member, setup

def simple_decorator(f):
    if False:
        for i in range(10):
            print('nop')
    '\n    A simple decorator that does nothing, for tests to use.\n    '

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return f(*args, **kwargs)
    return wrapper

def _private_doc():
    if False:
        return 10
    'module._private_doc.DOCSTRING'
    pass

def _private_undoc():
    if False:
        while True:
            i = 10
    pass

def __special_doc__():
    if False:
        i = 10
        return i + 15
    'module.__special_doc__.DOCSTRING'
    pass

def __special_undoc__():
    if False:
        print('Hello World!')
    pass

class SampleClass:

    def _private_doc(self):
        if False:
            i = 10
            return i + 15
        'SampleClass._private_doc.DOCSTRING'
        pass

    def _private_undoc(self):
        if False:
            while True:
                i = 10
        pass

    def __special_doc__(self):
        if False:
            i = 10
            return i + 15
        'SampleClass.__special_doc__.DOCSTRING'
        pass

    def __special_undoc__(self):
        if False:
            while True:
                i = 10
        pass

    @simple_decorator
    def __decorated_func__(self):
        if False:
            for i in range(10):
                print('nop')
        'doc'
        pass

class SampleError(Exception):

    def _private_doc(self):
        if False:
            for i in range(10):
                print('nop')
        'SampleError._private_doc.DOCSTRING'
        pass

    def _private_undoc(self):
        if False:
            while True:
                i = 10
        pass

    def __special_doc__(self):
        if False:
            print('Hello World!')
        'SampleError.__special_doc__.DOCSTRING'
        pass

    def __special_undoc__(self):
        if False:
            for i in range(10):
                print('nop')
        pass
SampleNamedTuple = namedtuple('SampleNamedTuple', 'user_id block_type def_id')

class TestProcessDocstring:

    def test_modify_in_place(self):
        if False:
            i = 10
            return i + 15
        lines = ['Summary line.', '', 'Args:', '   arg1: arg1 description']
        app = mock.Mock()
        app.config = Config()
        _process_docstring(app, 'class', 'SampleClass', SampleClass, mock.Mock(), lines)
        expected = ['Summary line.', '', ':param arg1: arg1 description', '']
        assert expected == lines

class TestSetup:

    def test_unknown_app_type(self):
        if False:
            return 10
        setup(object())

    def test_add_config_values(self):
        if False:
            return 10
        app = mock.Mock(Sphinx)
        setup(app)
        for name in Config._config_values:
            has_config = False
            for (method_name, args, _kwargs) in app.method_calls:
                if method_name == 'add_config_value' and args[0] == name:
                    has_config = True
            if not has_config:
                pytest.fail('Config value was not added to app %s' % name)
        has_process_docstring = False
        has_skip_member = False
        for (method_name, args, _kwargs) in app.method_calls:
            if method_name == 'connect':
                if args[0] == 'autodoc-process-docstring' and args[1] == _process_docstring:
                    has_process_docstring = True
                elif args[0] == 'autodoc-skip-member' and args[1] == _skip_member:
                    has_skip_member = True
        if not has_process_docstring:
            pytest.fail('autodoc-process-docstring never connected')
        if not has_skip_member:
            pytest.fail('autodoc-skip-member never connected')

class TestSkipMember:

    def assert_skip(self, what, member, obj, expect_default_skip, config_name):
        if False:
            for i in range(10):
                print('nop')
        skip = True
        app = mock.Mock()
        app.config = Config()
        setattr(app.config, config_name, True)
        if expect_default_skip:
            assert None is _skip_member(app, what, member, obj, skip, mock.Mock())
        else:
            assert _skip_member(app, what, member, obj, skip, mock.Mock()) is False
        setattr(app.config, config_name, False)
        assert None is _skip_member(app, what, member, obj, skip, mock.Mock())

    def test_namedtuple(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_skip('class', '_asdict', SampleNamedTuple._asdict, True, 'napoleon_include_private_with_doc')

    def test_class_private_doc(self):
        if False:
            print('Hello World!')
        self.assert_skip('class', '_private_doc', SampleClass._private_doc, False, 'napoleon_include_private_with_doc')

    def test_class_private_undoc(self):
        if False:
            i = 10
            return i + 15
        self.assert_skip('class', '_private_undoc', SampleClass._private_undoc, True, 'napoleon_include_private_with_doc')

    def test_class_special_doc(self):
        if False:
            i = 10
            return i + 15
        self.assert_skip('class', '__special_doc__', SampleClass.__special_doc__, False, 'napoleon_include_special_with_doc')

    def test_class_special_undoc(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_skip('class', '__special_undoc__', SampleClass.__special_undoc__, True, 'napoleon_include_special_with_doc')

    def test_class_decorated_doc(self):
        if False:
            while True:
                i = 10
        self.assert_skip('class', '__decorated_func__', SampleClass.__decorated_func__, False, 'napoleon_include_special_with_doc')

    def test_exception_private_doc(self):
        if False:
            i = 10
            return i + 15
        self.assert_skip('exception', '_private_doc', SampleError._private_doc, False, 'napoleon_include_private_with_doc')

    def test_exception_private_undoc(self):
        if False:
            for i in range(10):
                print('nop')
        self.assert_skip('exception', '_private_undoc', SampleError._private_undoc, True, 'napoleon_include_private_with_doc')

    def test_exception_special_doc(self):
        if False:
            return 10
        self.assert_skip('exception', '__special_doc__', SampleError.__special_doc__, False, 'napoleon_include_special_with_doc')

    def test_exception_special_undoc(self):
        if False:
            print('Hello World!')
        self.assert_skip('exception', '__special_undoc__', SampleError.__special_undoc__, True, 'napoleon_include_special_with_doc')

    def test_module_private_doc(self):
        if False:
            while True:
                i = 10
        self.assert_skip('module', '_private_doc', _private_doc, False, 'napoleon_include_private_with_doc')

    def test_module_private_undoc(self):
        if False:
            i = 10
            return i + 15
        self.assert_skip('module', '_private_undoc', _private_undoc, True, 'napoleon_include_private_with_doc')

    def test_module_special_doc(self):
        if False:
            while True:
                i = 10
        self.assert_skip('module', '__special_doc__', __special_doc__, False, 'napoleon_include_special_with_doc')

    def test_module_special_undoc(self):
        if False:
            return 10
        self.assert_skip('module', '__special_undoc__', __special_undoc__, True, 'napoleon_include_special_with_doc')