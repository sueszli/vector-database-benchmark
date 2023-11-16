from __future__ import annotations
from typing import Dict, Generic, Iterable, TypeVar
from typing_extensions import assert_type
good: dict[str, str] = dict()
assert_type(good, Dict[str, str])
assert_type(dict(arg=1), Dict[str, int])
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')

class KeysAndGetItem(Generic[_KT, _VT]):
    data: dict[_KT, _VT]

    def __init__(self, data: dict[_KT, _VT]) -> None:
        if False:
            print('Hello World!')
        self.data = data

    def keys(self) -> Iterable[_KT]:
        if False:
            print('Hello World!')
        return self.data.keys()

    def __getitem__(self, __k: _KT) -> _VT:
        if False:
            return 10
        return self.data[__k]
kt1: KeysAndGetItem[int, str] = KeysAndGetItem({0: ''})
assert_type(dict(kt1), Dict[int, str])
dict(kt1, arg='a')
kt2: KeysAndGetItem[str, int] = KeysAndGetItem({'': 0})
assert_type(dict(kt2, arg=1), Dict[str, int])

def test_iterable_tuple_overload(x: Iterable[tuple[int, str]]) -> dict[int, str]:
    if False:
        for i in range(10):
            print('nop')
    return dict(x)
i1: Iterable[tuple[int, str]] = [(1, 'a'), (2, 'b')]
test_iterable_tuple_overload(i1)
dict(i1, arg='a')
i2: Iterable[tuple[str, int]] = [('a', 1), ('b', 2)]
assert_type(dict(i2, arg=1), Dict[str, int])
i3: Iterable[str] = ['a.b']
i4: Iterable[bytes] = [b'a.b']
assert_type(dict((string.split('.') for string in i3)), Dict[str, str])
assert_type(dict((string.split(b'.') for string in i4)), Dict[bytes, bytes])
dict(['foo', 'bar', 'baz'])
dict([b'foo', b'bar', b'baz'])