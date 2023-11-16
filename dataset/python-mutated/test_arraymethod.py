"""
This file tests the generic aspects of ArrayMethod.  At the time of writing
this is private API, but when added, public API may be added here.
"""
from __future__ import annotations
import sys
import types
from typing import Any
import pytest
import numpy as np
from numpy._core._multiarray_umath import _get_castingimpl as get_castingimpl

class TestResolveDescriptors:
    method = get_castingimpl(type(np.dtype('d')), type(np.dtype('f')))

    @pytest.mark.parametrize('args', [(True,), (None,), ((None, None, None),), ((None, None),), ((np.dtype('d'), True),), ((np.dtype('f'), None),)])
    def test_invalid_arguments(self, args):
        if False:
            return 10
        with pytest.raises(TypeError):
            self.method._resolve_descriptors(*args)

class TestSimpleStridedCall:
    method = get_castingimpl(type(np.dtype('d')), type(np.dtype('f')))

    @pytest.mark.parametrize(['args', 'error'], [((True,), TypeError), (((None,),), TypeError), ((None, None), TypeError), (((None, None, None),), TypeError), (((np.arange(3), np.arange(3)),), TypeError), (((np.ones(3, dtype='>d'), np.ones(3, dtype='<f')),), TypeError), (((np.ones((2, 2), dtype='d'), np.ones((2, 2), dtype='f')),), ValueError), (((np.ones(3, dtype='d'), np.ones(4, dtype='f')),), ValueError), (((np.frombuffer(b'\x00x00' * 3 * 2, dtype='d'), np.frombuffer(b'\x00x00' * 3, dtype='f')),), ValueError)])
    def test_invalid_arguments(self, args, error):
        if False:
            print('Hello World!')
        with pytest.raises(error):
            self.method._simple_strided_call(*args)

@pytest.mark.parametrize('cls', [np.ndarray, np.recarray, np.char.chararray, np.matrix, np.memmap])
class TestClassGetItem:

    def test_class_getitem(self, cls: type[np.ndarray]) -> None:
        if False:
            i = 10
            return i + 15
        'Test `ndarray.__class_getitem__`.'
        alias = cls[Any, Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is cls

    @pytest.mark.parametrize('arg_len', range(4))
    def test_subscript_tup(self, cls: type[np.ndarray], arg_len: int) -> None:
        if False:
            while True:
                i = 10
        arg_tup = (Any,) * arg_len
        if arg_len in (1, 2):
            assert cls[arg_tup]
        else:
            match = f"Too {('few' if arg_len == 0 else 'many')} arguments"
            with pytest.raises(TypeError, match=match):
                cls[arg_tup]