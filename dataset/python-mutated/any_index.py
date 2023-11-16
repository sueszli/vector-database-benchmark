from builtins import _test_sink, _test_source
from typing import Dict, Optional

def post_0(i: int) -> str:
    if False:
        return 10
    taint = {}
    taint[i] = _test_source()
    taint[0] = _test_source()
    return taint[0]

def post_1(i: int) -> str:
    if False:
        return 10
    taint = {}
    taint[0] = _test_source()
    taint[i] = _test_source()
    return taint[0]

def post_2(i: int) -> str:
    if False:
        while True:
            i = 10
    taint = {}
    if 1 > 2:
        taint[i] = _test_source()
    else:
        taint[0] = _test_source()
    return taint[0]

def post_3(i: int) -> str:
    if False:
        while True:
            i = 10
    taint = {}
    if 1 > 2:
        taint[0] = _test_source()
    else:
        taint[i] = _test_source()
    return taint[0]

def pre_0(x: str, i: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    taint = {}
    taint[0] = x
    _test_sink(taint[i])
    _test_sink(taint[0])

def pre_1(x: str, i: int) -> None:
    if False:
        return 10
    taint = {}
    taint[0] = x
    _test_sink(taint[0])
    _test_sink(taint[i])

def pre_2(x: str, i: int) -> None:
    if False:
        return 10
    taint = {}
    taint[0] = x
    if 1 > 2:
        _test_sink(taint[i])
    else:
        _test_sink(taint[0])

def pre_3(x: str, i: int) -> None:
    if False:
        while True:
            i = 10
    taint = {}
    taint[0] = x
    if 1 > 2:
        _test_sink(taint[0])
    else:
        _test_sink(taint[i])