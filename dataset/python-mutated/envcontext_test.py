from __future__ import annotations
import os
from unittest import mock
import pytest
from pre_commit.envcontext import envcontext
from pre_commit.envcontext import UNSET
from pre_commit.envcontext import Var

def _test(*, before, patch, expected):
    if False:
        print('Hello World!')
    env = before.copy()
    with envcontext(patch, _env=env):
        assert env == expected
    assert env == before

def test_trivial():
    if False:
        while True:
            i = 10
    _test(before={}, patch={}, expected={})

def test_noop():
    if False:
        i = 10
        return i + 15
    _test(before={'foo': 'bar'}, patch=(), expected={'foo': 'bar'})

def test_adds():
    if False:
        i = 10
        return i + 15
    _test(before={}, patch=[('foo', 'bar')], expected={'foo': 'bar'})

def test_overrides():
    if False:
        i = 10
        return i + 15
    _test(before={'foo': 'baz'}, patch=[('foo', 'bar')], expected={'foo': 'bar'})

def test_unset_but_nothing_to_unset():
    if False:
        return 10
    _test(before={}, patch=[('foo', UNSET)], expected={})

def test_unset_things_to_remove():
    if False:
        print('Hello World!')
    _test(before={'PYTHONHOME': ''}, patch=[('PYTHONHOME', UNSET)], expected={})

def test_templated_environment_variable_missing():
    if False:
        print('Hello World!')
    _test(before={}, patch=[('PATH', ('~/bin:', Var('PATH')))], expected={'PATH': '~/bin:'})

def test_templated_environment_variable_defaults():
    if False:
        i = 10
        return i + 15
    _test(before={}, patch=[('PATH', ('~/bin:', Var('PATH', default='/bin')))], expected={'PATH': '~/bin:/bin'})

def test_templated_environment_variable_there():
    if False:
        return 10
    _test(before={'PATH': '/usr/local/bin:/usr/bin'}, patch=[('PATH', ('~/bin:', Var('PATH')))], expected={'PATH': '~/bin:/usr/local/bin:/usr/bin'})

def test_templated_environ_sources_from_previous():
    if False:
        for i in range(10):
            print('nop')
    _test(before={'foo': 'bar'}, patch=(('foo', 'baz'), ('herp', ('foo: ', Var('foo')))), expected={'foo': 'baz', 'herp': 'foo: bar'})

def test_exception_safety():
    if False:
        while True:
            i = 10

    class MyError(RuntimeError):
        pass
    env = {'hello': 'world'}
    with pytest.raises(MyError):
        with envcontext((('foo', 'bar'),), _env=env):
            raise MyError()
    assert env == {'hello': 'world'}

def test_integration_os_environ():
    if False:
        return 10
    with mock.patch.dict(os.environ, {'FOO': 'bar'}, clear=True):
        assert os.environ == {'FOO': 'bar'}
        with envcontext((('HERP', 'derp'),)):
            assert os.environ == {'FOO': 'bar', 'HERP': 'derp'}
        assert os.environ == {'FOO': 'bar'}