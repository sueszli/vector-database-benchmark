from collections import defaultdict
import pytest
from funcy.funcmakers import *

def test_callable():
    if False:
        for i in range(10):
            print('nop')
    assert make_func(lambda x: x + 42)(0) == 42

def test_int():
    if False:
        i = 10
        return i + 15
    assert make_func(0)('abc') == 'a'
    assert make_func(2)([1, 2, 3]) == 3
    assert make_func(1)({1: 'a'}) == 'a'
    with pytest.raises(IndexError):
        make_func(1)('a')
    with pytest.raises(TypeError):
        make_func(1)(42)

def test_slice():
    if False:
        print('Hello World!')
    assert make_func(slice(1, None))('abc') == 'bc'

def test_str():
    if False:
        return 10
    assert make_func('\\d+')('ab42c') == '42'
    assert make_func('\\d+')('abc') is None
    assert make_pred('\\d+')('ab42c') is True
    assert make_pred('\\d+')('abc') is False

def test_dict():
    if False:
        for i in range(10):
            print('nop')
    assert make_func({1: 'a'})(1) == 'a'
    with pytest.raises(KeyError):
        make_func({1: 'a'})(2)
    d = defaultdict(int, a=42)
    assert make_func(d)('a') == 42
    assert make_func(d)('b') == 0

def test_set():
    if False:
        for i in range(10):
            print('nop')
    s = set([1, 2, 3])
    assert make_func(s)(1) is True
    assert make_func(s)(4) is False