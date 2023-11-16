from builtins import _test_sink, _test_source
from typing import Dict

def sink_on_0(x: Dict) -> None:
    if False:
        print('Hello World!')
    _test_sink(x['0'])

def sink_on_0_and_star(x: Dict, i: int) -> None:
    if False:
        i = 10
        return i + 15
    _test_sink(x['0'])
    _test_sink(x[i])

def issue_source_on_0_and_star_to_sink_on_0_and_star(i: int) -> None:
    if False:
        while True:
            i = 10
    x = {}
    x[i] = _test_source()
    x['0'] = _test_source()
    sink_on_0_and_star(x, i)

def issue_source_on_0_to_sink_on_0_and_star(i: int) -> None:
    if False:
        return 10
    x = {}
    x['0'] = _test_source()
    sink_on_0_and_star(x, i)

def issue_source_on_0_and_star_to_sink_on_0(i: int) -> None:
    if False:
        return 10
    x = {}
    x[i] = _test_source()
    x['0'] = _test_source()
    sink_on_0(x)

def issue_source_on_0_to_sink_on_0() -> None:
    if False:
        for i in range(10):
            print('nop')
    x = {}
    x['0'] = _test_source()
    sink_on_0(x)

def issue_source_on_1_to_sink_on_0_and_star(i: int) -> None:
    if False:
        print('Hello World!')
    x = {}
    x['1'] = _test_source()
    sink_on_0_and_star(x, i)

def no_issue_source_on_1_to_sink_on_0() -> None:
    if False:
        for i in range(10):
            print('nop')
    x = {}
    x['1'] = _test_source()
    sink_on_0(x)

def issue_source_on_star_to_sink_on_0_and_star(i: int) -> None:
    if False:
        print('Hello World!')
    x = {}
    x[i] = _test_source()
    sink_on_0_and_star(x, i)

def issue_source_on_star_to_sink_on_0(i: int) -> None:
    if False:
        print('Hello World!')
    x = {}
    x[i] = _test_source()
    sink_on_0(x)