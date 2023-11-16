from __future__ import annotations
import datetime
import enum
from dataclasses import dataclass
from importlib import import_module
from typing import ClassVar
import attr
import pytest
from pydantic import BaseModel
from airflow.datasets import Dataset
from airflow.serialization.serde import CLASSNAME, DATA, SCHEMA_ID, VERSION, _get_patterns, _match, deserialize, serialize
from airflow.utils.module_loading import import_string, iter_namespace, qualname
from tests.test_utils.config import conf_vars

@pytest.fixture()
def recalculate_patterns():
    if False:
        for i in range(10):
            print('nop')
    _get_patterns.cache_clear()

class Z:
    __version__: ClassVar[int] = 1

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.x = x

    def serialize(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return dict({'x': self.x})

    @staticmethod
    def deserialize(data: dict, version: int):
        if False:
            print('Hello World!')
        if version != 1:
            raise TypeError('version != 1')
        return Z(data['x'])

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.x == other.x

@attr.define
class Y:
    x: int
    __version__: ClassVar[int] = 1

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.x = x

class X:
    pass

@dataclass
class W:
    __version__: ClassVar[int] = 2
    x: int

@dataclass
class V:
    __version__: ClassVar[int] = 1
    w: W
    s: list
    t: tuple
    c: int

class U(BaseModel):
    __version__: ClassVar[int] = 1
    x: int
    v: V
    u: tuple

class C:

    def __call__(self):
        if False:
            return 10
        return None

@pytest.mark.usefixtures('recalculate_patterns')
class TestSerDe:

    def test_ser_primitives(self):
        if False:
            print('Hello World!')
        i = 10
        e = serialize(i)
        assert i == e
        i = 10.1
        e = serialize(i)
        assert i == e
        i = 'test'
        e = serialize(i)
        assert i == e
        i = True
        e = serialize(i)
        assert i == e
        Color = enum.IntEnum('Color', ['RED', 'GREEN'])
        i = Color.RED
        e = serialize(i)
        assert i == e

    def test_ser_collections(self):
        if False:
            i = 10
            return i + 15
        i = [1, 2]
        e = deserialize(serialize(i))
        assert i == e
        i = ('a', 'b', 'a', 'c')
        e = deserialize(serialize(i))
        assert i == e
        i = {2, 3}
        e = deserialize(serialize(i))
        assert i == e
        i = frozenset({6, 7})
        e = deserialize(serialize(i))
        assert i == e

    def test_der_collections_compat(self):
        if False:
            for i in range(10):
                print('nop')
        i = [1, 2]
        e = deserialize(i)
        assert i == e
        i = ('a', 'b', 'a', 'c')
        e = deserialize(i)
        assert i == e
        i = {2, 3}
        e = deserialize(i)
        assert i == e

    def test_ser_plain_dict(self):
        if False:
            i = 10
            return i + 15
        i = {'a': 1, 'b': 2}
        e = serialize(i)
        assert i == e
        with pytest.raises(AttributeError, match='^reserved'):
            i = {CLASSNAME: 'cannot'}
            serialize(i)
        with pytest.raises(AttributeError, match='^reserved'):
            i = {SCHEMA_ID: 'cannot'}
            serialize(i)

    def test_no_serializer(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(TypeError, match='^cannot serialize'):
            i = Exception
            serialize(i)

    def test_ser_registered(self):
        if False:
            while True:
                i = 10
        i = datetime.datetime(2000, 10, 1)
        e = serialize(i)
        assert e[DATA]

    def test_serder_custom(self):
        if False:
            print('Hello World!')
        i = Z(1)
        e = serialize(i)
        assert Z.__version__ == e[VERSION]
        assert qualname(Z) == e[CLASSNAME]
        assert e[DATA]
        d = deserialize(e)
        assert i.x == getattr(d, 'x', None)

    def test_serder_attr(self):
        if False:
            print('Hello World!')
        i = Y(10)
        e = serialize(i)
        assert Y.__version__ == e[VERSION]
        assert qualname(Y) == e[CLASSNAME]
        assert e[DATA]
        d = deserialize(e)
        assert i.x == getattr(d, 'x', None)

    def test_serder_dataclass(self):
        if False:
            i = 10
            return i + 15
        i = W(12)
        e = serialize(i)
        assert W.__version__ == e[VERSION]
        assert qualname(W) == e[CLASSNAME]
        assert e[DATA]
        d = deserialize(e)
        assert i.x == getattr(d, 'x', None)

    @conf_vars({('core', 'allowed_deserialization_classes'): 'airflow[.].*'})
    @pytest.mark.usefixtures('recalculate_patterns')
    def test_allow_list_for_imports(self):
        if False:
            print('Hello World!')
        i = Z(10)
        e = serialize(i)
        with pytest.raises(ImportError) as ex:
            deserialize(e)
        assert f'{qualname(Z)} was not found in allow list' in str(ex.value)

    @conf_vars({('core', 'allowed_deserialization_classes'): 'tests.*'})
    @pytest.mark.usefixtures('recalculate_patterns')
    def test_allow_list_replace(self):
        if False:
            for i in range(10):
                print('nop')
        assert _match('tests.airflow.deep')
        assert _match('testsfault') is False

    def test_incompatible_version(self):
        if False:
            for i in range(10):
                print('nop')
        data = dict({'__classname__': Y.__module__ + '.' + Y.__qualname__, '__version__': 2})
        with pytest.raises(TypeError, match='newer than'):
            deserialize(data)

    def test_raise_undeserializable(self):
        if False:
            return 10
        data = dict({'__classname__': X.__module__ + '.' + X.__qualname__, '__version__': 0})
        with pytest.raises(TypeError, match='No deserializer'):
            deserialize(data)

    def test_backwards_compat(self):
        if False:
            print('Hello World!')
        '\n        Verify deserialization of old-style encoded Xcom values including nested ones\n        '
        uri = 's3://does_not_exist'
        data = {'__type': 'airflow.datasets.Dataset', '__source': None, '__var': {'__var': {'uri': uri, 'extra': {'__var': {'hi': 'bye'}, '__type': 'dict'}}, '__type': 'dict'}}
        dataset = deserialize(data)
        assert dataset.extra == {'hi': 'bye'}
        assert dataset.uri == uri

    def test_backwards_compat_wrapped(self):
        if False:
            return 10
        '\n        Verify deserialization of old-style wrapped XCom value\n        '
        i = {'extra': {'__var': {'hi': 'bye'}, '__type': 'dict'}}
        e = deserialize(i)
        assert e['extra'] == {'hi': 'bye'}

    def test_encode_dataset(self):
        if False:
            return 10
        dataset = Dataset('mytest://dataset')
        obj = deserialize(serialize(dataset))
        assert dataset.uri == obj.uri

    def test_serializers_importable_and_str(self):
        if False:
            for i in range(10):
                print('nop')
        'test if all distributed serializers are lazy loading and can be imported'
        import airflow.serialization.serializers
        for (_, name, _) in iter_namespace(airflow.serialization.serializers):
            mod = import_module(name)
            for s in getattr(mod, 'serializers', list()):
                if not isinstance(s, str):
                    raise TypeError(f'{s} is not of type str. This is required for lazy loading')
                try:
                    import_string(s)
                except ImportError:
                    raise AttributeError(f'{s} cannot be imported (located in {name})')

    def test_stringify(self):
        if False:
            while True:
                i = 10
        i = V(W(10), ['l1', 'l2'], (1, 2), 10)
        e = serialize(i)
        s = deserialize(e, full=False)
        assert f'{qualname(V)}@version={V.__version__}' in s
        assert "w={'x': 10}" in s
        assert "s=['l1', 'l2']" in s
        assert 't=(1,2)' in s
        assert 'c=10' in s
        e['__data__']['t'] = (1, 2)
        s = deserialize(e, full=False)

    @pytest.mark.parametrize('obj, expected', [(Z(10), {'__classname__': 'tests.serialization.test_serde.Z', '__version__': 1, '__data__': {'x': 10}}), (W(2), {'__classname__': 'tests.serialization.test_serde.W', '__version__': 2, '__data__': {'x': 2}})])
    def test_serialized_data(self, obj, expected):
        if False:
            return 10
        assert expected == serialize(obj)

    def test_deserialize_non_serialized_data(self):
        if False:
            return 10
        i = Z(10)
        e = deserialize(i)
        assert i == e

    def test_pydantic(self):
        if False:
            return 10
        i = U(x=10, v=V(W(10), ['l1', 'l2'], (1, 2), 10), u=(1, 2))
        e = serialize(i)
        s = deserialize(e)
        assert i == s

    def test_error_when_serializing_callable_without_name(self):
        if False:
            while True:
                i = 10
        i = C()
        with pytest.raises(TypeError, match="cannot serialize object of type <class 'tests.serialization.test_serde.C'>"):
            serialize(i)