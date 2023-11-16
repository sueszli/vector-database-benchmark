import inspect
import json
import sys
import tempfile
from functools import update_wrapper
import pytest
from dagster import DagsterType, _seven
from dagster._core.types.dagster_type import ListType
from dagster._seven import is_subclass
from dagster._utils import file_relative_path

def test_is_ascii():
    if False:
        while True:
            i = 10
    assert _seven.is_ascii('Hello!')
    assert not _seven.is_ascii('您好!')

def test_import_module_from_path():
    if False:
        print('Hello World!')
    foo_module = _seven.import_module_from_path('foo_module', file_relative_path(__file__, 'foo_module.py'))
    assert foo_module.FOO == 7

def test_json_decode_error():
    if False:
        print('Hello World!')
    with pytest.raises(_seven.json.JSONDecodeError):
        json.loads(',dsfjd')

def test_json_dump():
    if False:
        for i in range(10):
            print('nop')
    with tempfile.TemporaryFile('w+') as fd:
        _seven.json.dump({'foo': 'bar', 'a': 'b'}, fd)
        fd.seek(0)
        assert fd.read() == '{"a": "b", "foo": "bar"}'

def test_json_dumps():
    if False:
        return 10
    assert _seven.json.dumps({'foo': 'bar', 'a': 'b'}) == '{"a": "b", "foo": "bar"}'

def test_tempdir():
    if False:
        print('Hello World!')
    assert not _seven.temp_dir.get_system_temp_directory().startswith('/var')

def test_get_arg_names():
    if False:
        for i in range(10):
            print('nop')

    def foo(one, two=2, three=None):
        if False:
            for i in range(10):
                print('nop')
        pass
    assert len(_seven.get_arg_names(foo)) == 3
    assert 'one' in _seven.get_arg_names(foo)
    assert 'two' in _seven.get_arg_names(foo)
    assert 'three' in _seven.get_arg_names(foo)

def test_is_lambda():
    if False:
        for i in range(10):
            print('nop')
    foo = lambda : None

    def bar():
        if False:
            print('Hello World!')
        pass
    baz = 3

    class Oof:
        test = lambda x: x
    assert _seven.is_lambda(foo) is True
    assert _seven.is_lambda(Oof.test) is True
    assert _seven.is_lambda(bar) is False
    assert _seven.is_lambda(baz) is False

def test_is_fn_or_decor_inst():
    if False:
        while True:
            i = 10

    class Quux:
        pass

    def foo():
        if False:
            i = 10
            return i + 15
        return Quux()
    bar = lambda _: Quux()
    baz = Quux()

    def quux_decor(fn):
        if False:
            for i in range(10):
                print('nop')
        q = Quux()
        return update_wrapper(q, fn)

    @quux_decor
    def yoodles():
        if False:
            print('Hello World!')
        pass
    assert _seven.is_function_or_decorator_instance_of(foo, Quux) is True
    assert _seven.is_function_or_decorator_instance_of(bar, Quux) is True
    assert _seven.is_function_or_decorator_instance_of(baz, Quux) is False
    assert _seven.is_function_or_decorator_instance_of(yoodles, Quux) is True

class Foo:
    pass

class Bar(Foo):
    pass

def test_is_subclass():
    if False:
        while True:
            i = 10
    assert is_subclass(Bar, Foo)
    assert not is_subclass(Foo, Bar)
    assert is_subclass(DagsterType, DagsterType)
    assert is_subclass(str, str)
    assert is_subclass(ListType, DagsterType)
    assert not is_subclass(DagsterType, ListType)
    assert not is_subclass(ListType, str)
    assert not inspect.isclass(2)
    assert not is_subclass(2, DagsterType)

@pytest.mark.skipif(sys.version_info.minor < 9, reason='Generic aliases only exist on py39 or later')
def test_is_subclass_generic_alias():
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError):
        issubclass(list[str], DagsterType)
    assert not is_subclass(list[str], DagsterType)