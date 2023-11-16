from __future__ import annotations
import dataclasses as dc
from typing import TYPE_CHECKING, Any, Dict, Tuple, Type, Union
from typing_extensions import assert_type
if TYPE_CHECKING:
    from _typeshed import DataclassInstance

@dc.dataclass
class Foo:
    attr: str
assert_type(dc.fields(Foo), Tuple[dc.Field[Any], ...])
f = Foo(attr='attr')
assert_type(dc.fields(f), Tuple[dc.Field[Any], ...])
assert_type(dc.asdict(f), Dict[str, Any])
assert_type(dc.astuple(f), Tuple[Any, ...])
assert_type(dc.replace(f, attr='new'), Foo)
if dc.is_dataclass(f):
    assert_type(f, Foo)

def check_other_isdataclass_overloads(x: type, y: object) -> None:
    if False:
        for i in range(10):
            print('nop')
    dc.fields(y)
    dc.asdict(x)
    dc.asdict(y)
    dc.astuple(x)
    dc.astuple(y)
    dc.replace(x)
    dc.replace(y)
    if dc.is_dataclass(x):
        assert_type(x, Type['DataclassInstance'])
        assert_type(dc.fields(x), Tuple[dc.Field[Any], ...])
    if dc.is_dataclass(y):
        assert_type(y, Union['DataclassInstance', Type['DataclassInstance']])
        assert_type(dc.fields(y), Tuple[dc.Field[Any], ...])
    if dc.is_dataclass(y) and (not isinstance(y, type)):
        assert_type(y, 'DataclassInstance')
        assert_type(dc.fields(y), Tuple[dc.Field[Any], ...])
        assert_type(dc.asdict(y), Dict[str, Any])
        assert_type(dc.astuple(y), Tuple[Any, ...])
        dc.replace(y)