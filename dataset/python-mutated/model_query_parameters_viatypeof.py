from builtins import _test_source
from typing import Annotated, Dict, List

def test1_f1(a, b, c):
    if False:
        i = 10
        return i + 15
    pass

def test1_alarm1():
    if False:
        print('Hello World!')
    x: str = _test_source()
    test1_f1(x, 'b', 0)

def test1_alarm2():
    if False:
        i = 10
        return i + 15
    x: Annotated[str, 'foo'] = _test_source()
    test1_f1('a', x, 0)

def test1_alarm3():
    if False:
        return 10
    x: int = _test_source()
    test1_f1('a', 'b', x)

def test1_noalarm1():
    if False:
        return 10
    test1_f1('a', 'b', 0)

class Test2_C:

    def f1(self, a, b, c):
        if False:
            return 10
        pass

    def f2(self, a, b, c):
        if False:
            i = 10
            return i + 15
        pass

class Test2_T:
    pass

def test2_alarm1(c: Test2_C):
    if False:
        i = 10
        return i + 15
    x: str = _test_source()
    c.f1(x, 'b', 0)

def test2_alarm2(c: Test2_C):
    if False:
        for i in range(10):
            print('nop')
    x: Dict[str, int] = _test_source()
    c.f1('a', x, 0)

def test2_alarm3(c: Test2_C):
    if False:
        return 10
    x: Test2_T = _test_source()
    c.f1('a', 'b', x)

def test2_alarm4(c: Test2_C):
    if False:
        while True:
            i = 10
    x: int = _test_source()
    c.f2(x, 'b', 0)

def test2_alarm5(c: Test2_C):
    if False:
        print('Hello World!')
    x: List[List[Test2_T]] = _test_source()
    c.f2('a', x, 0)

def test2_alarm6(c: Test2_C):
    if False:
        for i in range(10):
            print('nop')
    x: float = _test_source()
    c.f2('a', 'b', x)

def test2_noalarm1(c: Test2_C):
    if False:
        i = 10
        return i + 15
    c.f1([], {}, 0.0)

def test2_noalarm2(c: Test2_C):
    if False:
        while True:
            i = 10
    c.f2('a', 0, Test2_T())

def test3_f1(a, b, c):
    if False:
        i = 10
        return i + 15
    pass

def test3_alarm1():
    if False:
        i = 10
        return i + 15
    x: str = _test_source()
    test3_f1(x, 'b', 0)

def test3_alarm2():
    if False:
        print('Hello World!')
    x: Annotated[str, 'foo'] = _test_source()
    test3_f1('a', x, 0)

def test3_alarm3():
    if False:
        i = 10
        return i + 15
    x: int = _test_source()
    test3_f1('a', 'b', x)

def test3_noalarm1():
    if False:
        for i in range(10):
            print('nop')
    test3_f1('a', 'b', 0)