from builtins import _test_sink, _test_source
from typing import TypedDict

class SimpleTypedDict(TypedDict):
    foo: int
    bar: str

def test_typed_dict_setitem():
    if False:
        return 10
    d: SimpleTypedDict = {'foo': 0, 'bar': ''}
    d['bar'] = _test_source()
    _test_sink(d['bar'])
    _test_sink(d['foo'])

def test_typed_dict_constructor():
    if False:
        i = 10
        return i + 15
    d = SimpleTypedDict(foo=0, bar=_test_source())
    _test_sink(d['bar'])
    _test_sink(d['foo'])
    d = SimpleTypedDict(foo=0, bar={'a': _test_source()})
    _test_sink(d['bar']['a'])
    _test_sink(d['bar']['b'])
    _test_sink(d['foo']['a'])
    _test_sink(d['foo']['b'])
    d = SimpleTypedDict({'foo': 0, 'bar': _test_source()})
    _test_sink(d['bar'])
    _test_sink(d['foo'])
    d = SimpleTypedDict({_test_source(): 0})
    _test_sink(d.keys())
    _test_sink(d['foo'])
    _test_sink(d['bar'])
    d = SimpleTypedDict([('foo', 0), ('bar', _test_source())])
    _test_sink(d['bar'])
    _test_sink(d['foo'])

class SanitizedFieldTypedDict(TypedDict):
    sanitized: str
    safe: str

class NestedTypedDict(TypedDict):
    genuine: int
    nested: SanitizedFieldTypedDict

def test_sanitize_field():
    if False:
        i = 10
        return i + 15
    d: NestedTypedDict = _test_source()
    _test_sink(d['genuine'])
    d: NestedTypedDict = _test_source()
    _test_sink(d['nested']['sanitized'])
    bar: NestedTypedDict = _test_source()
    _test_sink(bar['nested']['safe'])