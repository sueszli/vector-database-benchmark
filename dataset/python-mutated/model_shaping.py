from builtins import _test_source, _rce, _sql, _user_controlled, _cookies
from typing import Any, cast, Dict, List

class RecordSchema:
    _META_PROP = ...

class DictRecord:
    items: Any = {}

class MutableRecord:
    __dict__: Dict[str, Any] = {}

def _is_dataclass_instance(obj) -> bool:
    if False:
        while True:
            i = 10
    ...

def fields(obj):
    if False:
        print('Hello World!')
    ...

def asdict(obj: RecordSchema, *, dict_factory: Any=dict) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    'Return the fields of a RecordSchema instance as a new dictionary mapping\n    field names to field values.\n    '
    return _asdict_inner(obj, dict_factory)

def _asdict_inner(obj: Any, dict_factory: Any) -> Any:
    if False:
        i = 10
        return i + 15
    meta = getattr(obj, RecordSchema._META_PROP, {})
    if _is_dataclass_instance(obj):
        result = []
        for f in fields(obj):
            value = _asdict_inner(getattr(obj, f.name), dict_factory)
            field_meta = meta.get(f.name)
            if value is not None or (field_meta and field_meta.include_none):
                name_override = field_meta and field_meta.name
                result.append((name_override or f.name, value))
        return dict_factory(result)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cast(List[Any], (_asdict_inner(v, dict_factory) for v in obj)))
    elif isinstance(obj, DictRecord):
        result = []
        for (k, v) in obj.items():
            value = _asdict_inner(v, dict_factory)
            field_meta = meta.get(k)
            if v is not None or (field_meta and field_meta.include_none):
                result.append((k, value))
        return dict_factory(result)
    elif isinstance(obj, MutableRecord):
        obj = obj.__dict__
        result = []
        for (k, v) in obj.items():
            value = _asdict_inner(v, dict_factory)
            field_meta = meta.get(k)
            if v is not None or (field_meta and field_meta.include_none):
                name_override = field_meta and field_meta.name
                result.append((name_override or k, value))
        return dict_factory(result)
    else:
        return obj

def asdict_test(obj):
    if False:
        return 10
    return asdict(obj)

def obscure_test(obj):
    if False:
        print('Hello World!')
    return type(obj)(_test_source())

def shape_multi_sink(obj):
    if False:
        while True:
            i = 10
    _rce(obj.foo)
    _rce(obj.bar)
    _rce(obj)
    _sql(obj.bar)

def shape_multi_source():
    if False:
        for i in range(10):
            print('nop')
    if 1 > 2:
        return {'a': _user_controlled(), 'a': {'b': _user_controlled()}, 'a': {'b': {'c': _user_controlled()}}}
    else:
        return {'a': {'b': _cookies()}}

def tito_shaping(parameters: Dict[str, Any]) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    return {'foo': parameters.get('foo'), 'bar': parameters.get('bar'), 'to_string': str(parameters)}

def test_tito_shaping() -> None:
    if False:
        return 10
    obj = tito_shaping({'foo': _test_source(), 'bar': {}})
    _test_sink(obj['foo'])
    _test_sink(obj['bar'])
    _test_sink(obj['to_string'])
    obj = tito_shaping({'foo': {'source': _test_source(), 'benign': ''}, 'bar': {}})
    _test_sink(obj['foo']['source'])
    _test_sink(obj['foo']['benign'])
    _test_sink(obj['bar'])
    _test_sink(obj['to_string'])