"""
Tests for `dict.__(r)or__`.

`dict.__or__` and `dict.__ror__` were only added in py39,
hence why these are in a separate file to the other test cases for `dict`.
"""
from __future__ import annotations
import os
import sys
from typing import Mapping, TypeVar, Union
from typing_extensions import Self, assert_type
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')
if sys.version_info >= (3, 9):

    class CustomDictSubclass(dict[_KT, _VT]):
        pass

    class CustomMappingWithDunderOr(Mapping[_KT, _VT]):

        def __or__(self, other: Mapping[_KT, _VT]) -> dict[_KT, _VT]:
            if False:
                while True:
                    i = 10
            return {}

        def __ror__(self, other: Mapping[_KT, _VT]) -> dict[_KT, _VT]:
            if False:
                i = 10
                return i + 15
            return {}

        def __ior__(self, other: Mapping[_KT, _VT]) -> Self:
            if False:
                for i in range(10):
                    print('nop')
            return self

    def test_dict_dot_or(a: dict[int, int], b: CustomDictSubclass[int, int], c: dict[str, str], d: Mapping[int, int], e: CustomMappingWithDunderOr[str, str]) -> None:
        if False:
            i = 10
            return i + 15
        assert_type(a | b, dict[int, int])
        assert_type(b | a, dict[int, int])
        assert_type(a | c, dict[Union[int, str], Union[int, str]])
        a | d
        assert_type(a | os.environ, dict[Union[str, int], Union[str, int]])
        assert_type(os.environ | a, dict[Union[str, int], Union[str, int]])
        assert_type(c | os.environ, dict[str, str])
        assert_type(c | e, dict[str, str])
        assert_type(os.environ | c, dict[str, str])
        assert_type(e | c, dict[str, str])
        e |= c
        e |= a
        c |= a