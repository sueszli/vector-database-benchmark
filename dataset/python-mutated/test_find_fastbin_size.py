from __future__ import annotations
import sys
from unittest.mock import MagicMock
import pytest
from pwnlib.util.packing import p64
module_name = 'pwndbg.commands'
module = MagicMock(__name__=module_name, load_commands=lambda : None)
sys.modules[module_name] = module
import mocks.gdb
import mocks.gdblib
from pwndbg.lib.heap.helpers import find_fastbin_size

def setup_mem(max_size, offsets):
    if False:
        i = 10
        return i + 15
    buf = bytearray([0] * max_size)
    for (offset, value) in offsets.items():
        buf[offset:offset + 8] = p64(value)
    return buf

def test_too_small():
    if False:
        i = 10
        return i + 15
    max_size = 128
    offsets = {8: 16}
    buf = setup_mem(max_size, offsets)
    with pytest.raises(StopIteration):
        next(find_fastbin_size(buf, max_size, 1))
    with pytest.raises(StopIteration):
        next(find_fastbin_size(buf, max_size, 8))

def test_normal():
    if False:
        for i in range(10):
            print('nop')
    max_size = 32
    offsets = {8: 32}
    buf = setup_mem(max_size, offsets)
    assert 0 == next(find_fastbin_size(buf, max_size, 1))
    assert 0 == next(find_fastbin_size(buf, max_size, 8))

def test_nozero_flags():
    if False:
        i = 10
        return i + 15
    max_size = 32
    offsets = {8: 47}
    buf = setup_mem(max_size, offsets)
    assert 0 == next(find_fastbin_size(buf, max_size, 1))
    assert 0 == next(find_fastbin_size(buf, max_size, 8))

def test_normal():
    if False:
        i = 10
        return i + 15
    max_size = 32
    offsets = {8: 32}
    buf = setup_mem(max_size, offsets)
    assert 0 == next(find_fastbin_size(buf, max_size, 1))
    assert 0 == next(find_fastbin_size(buf, max_size, 8))

def test_unaligned():
    if False:
        for i in range(10):
            print('nop')
    max_size = 32
    offsets = {9: 32}
    buf = setup_mem(max_size, offsets)
    assert 1 == next(find_fastbin_size(buf, max_size, 1))
    with pytest.raises(StopIteration):
        next(find_fastbin_size(buf, max_size, 8))