"""
Tests for `defaultdict.__or__` and `defaultdict.__ror__`.
These methods were only added in py39.
"""
from __future__ import annotations
import os
import sys
from collections import defaultdict
from typing import Mapping, TypeVar, Union
from typing_extensions import Self, assert_type
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')
if sys.version_info >= (3, 9):

    class CustomDefaultDictSubclass(defaultdict[_KT, _VT]):
        pass

    class CustomMappingWithDunderOr(Mapping[_KT, _VT]):

        def __or__(self, other: Mapping[_KT, _VT]) -> dict[_KT, _VT]:
            if False:
                return 10
            return {}

        def __ror__(self, other: Mapping[_KT, _VT]) -> dict[_KT, _VT]:
            if False:
                while True:
                    i = 10
            return {}

        def __ior__(self, other: Mapping[_KT, _VT]) -> Self:
            if False:
                return 10
            return self

    def test_defaultdict_dot_or(a: defaultdict[int, int], b: CustomDefaultDictSubclass[int, int], c: defaultdict[str, str], d: Mapping[int, int], e: CustomMappingWithDunderOr[str, str]) -> None:
        if False:
            i = 10
            return i + 15
        assert_type(a | b, defaultdict[int, int])
        assert_type(b | a, CustomDefaultDictSubclass[int, int])
        assert_type(a | c, defaultdict[Union[int, str], Union[int, str]])
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