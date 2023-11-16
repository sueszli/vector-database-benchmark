import enum
from builtins import _test_sink, _test_source
from typing import Annotated, Any, Dict, List
from dataclasses import dataclass

class Test1_C:

    def __init__(self, x: int, y: str, z: str) -> None:
        if False:
            while True:
                i = 10
        self.x: int = x
        self.y: str = y
        self.z: Annotated[str, 'test1'] = z

def test1_alarm1():
    if False:
        for i in range(10):
            print('nop')
    c = Test1_C(**_test_source())
    _test_sink(c.x)

def test1_alarm2():
    if False:
        for i in range(10):
            print('nop')
    c = Test1_C(**_test_source())
    _test_sink(c.y)

def test1_alarm3():
    if False:
        print('Hello World!')
    c = Test1_C(**_test_source())
    _test_sink(c.z)

def test1_alarm4(foo):
    if False:
        for i in range(10):
            print('nop')
    c = Test1_C(**_test_source())
    foo = c.x
    if 1:
        foo = c.y
    elif 2:
        foo = c.z
    _test_sink(foo)

@dataclass
class Test2_C:
    x: Dict[str, int] = {}
    y: List[str] = []
    z: Annotated[float, 'test2'] = 0.0

def test2_alarm1():
    if False:
        for i in range(10):
            print('nop')
    c = Test2_C(**_test_source())
    _test_sink(c.x)

def test2_alarm2():
    if False:
        return 10
    c = Test2_C(**_test_source())
    _test_sink(c.y)

def test2_alarm3():
    if False:
        i = 10
        return i + 15
    c = Test2_C(**_test_source())
    _test_sink(c.z)

def test2_alarm4(foo):
    if False:
        print('Hello World!')
    c = Test2_C(**_test_source())
    foo = c.x
    if 1:
        foo = c.y
    elif 2:
        foo = c.z
    _test_sink(foo)

def test2_alarm5_via_constructor():
    if False:
        while True:
            i = 10
    taint: str = _test_source()
    _test_sink(Test2_C(x={}, y=[], z=taint))

class Test3_Foo:
    ...

@dataclass
class Test3_C:
    x: Dict[str, List[int]] = {}
    y: Test3_Foo = Test3_Foo()
    z: Annotated[List[List[str]], 'test3'] = []

def test3_alarm1(c: Test3_C):
    if False:
        for i in range(10):
            print('nop')
    _test_sink(c.x)

def test3_alarm2(c: Test3_C):
    if False:
        while True:
            i = 10
    _test_sink(c.y)

def test3_alarm3(c: Test3_C):
    if False:
        i = 10
        return i + 15
    _test_sink(c.z)

def test3_alarm4(c: Test3_C):
    if False:
        return 10
    foo = c.x
    if 1:
        foo = c.y
    elif 2:
        foo = c.z
    _test_sink(foo)

@dataclass
class Test4_C:
    x = ...
    y: Any = 0
    z: object = []

def test4_alarm1(c: Test4_C):
    if False:
        while True:
            i = 10
    c.x = _test_source()

def test4_alarm2(c: Test4_C):
    if False:
        for i in range(10):
            print('nop')
    c.y = _test_source()

def test4_alarm3(c: Test4_C):
    if False:
        for i in range(10):
            print('nop')
    c.z = _test_source()

def return_via_parameter_type(parameter):
    if False:
        print('Hello World!')
    return 0

def test_strings():
    if False:
        i = 10
        return i + 15
    return return_via_parameter_type('A')

def test_numerals():
    if False:
        while True:
            i = 10
    return return_via_parameter_type(1)

def test_lists():
    if False:
        for i in range(10):
            print('nop')
    return return_via_parameter_type(['a', 'b'])

def meta(parameter):
    if False:
        while True:
            i = 10
    return return_via_parameter_type(parameter)

def test_via_type_of_does_not_propagate():
    if False:
        for i in range(10):
            print('nop')
    return meta('Name')

def tito(parameter, other):
    if False:
        i = 10
        return i + 15
    pass

def test_tito():
    if False:
        print('Hello World!')
    a = tito(_test_source(), [1, 2])
    return a

def sink_via_type_of(x, y):
    if False:
        while True:
            i = 10
    pass

def test_sink(element):
    if False:
        return 10
    return sink_via_type_of(element, 1)

def test_backwards_tito(parameter):
    if False:
        while True:
            i = 10
    return tito(parameter, 'by_backwards')