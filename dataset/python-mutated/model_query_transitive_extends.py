from builtins import _test_sink, _test_source

class Test1_C:
    attribute = ...

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.instance = ...

class Test1_C1(Test1_C):
    attribute = ...

    def __init__(self):
        if False:
            return 10
        self.instance = ...

class Test1_C2(Test1_C1):
    attribute = ...

    def __init__(self):
        if False:
            while True:
                i = 10
        self.instance = ...

class Test1_D:
    attribute = ...

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.instance = ...

class Test2_C:

    def foo(self, attribute):
        if False:
            while True:
                i = 10
        ...

class Test2_C1(Test2_C):

    def foo(self, attribute):
        if False:
            for i in range(10):
                print('nop')
        ...

class Test2_C2(Test2_C1):

    def foo(self, attribute):
        if False:
            return 10
        ...

class Test2_D:

    def foo(self, attribute):
        if False:
            for i in range(10):
                print('nop')
        ...

class UnrelatedClass:
    attribute = ...

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.instance = ...

    def foo(self, x):
        if False:
            return 10
        ...

def test1_alarm1(c: Test1_C1):
    if False:
        i = 10
        return i + 15
    _test_sink(c.attribute)

def test1_alarm2(c: Test1_C1):
    if False:
        for i in range(10):
            print('nop')
    _test_sink(c.instance)

def test1_alarm3(c: Test1_C2):
    if False:
        return 10
    _test_sink(c.attribute)

def test1_alarm4(c: Test1_C2):
    if False:
        return 10
    _test_sink(c.instance)

def test1_alarm5(c: Test1_C):
    if False:
        print('Hello World!')
    _test_sink(c.attribute)

def test1_alarm6(c: Test1_C):
    if False:
        while True:
            i = 10
    _test_sink(c.instance)

def test1_noalarm1(c: Test1_D):
    if False:
        while True:
            i = 10
    _test_sink(c.attribute)

def test1_noalarm2(c: Test1_D):
    if False:
        while True:
            i = 10
    _test_sink(c.instance)

def test2_alarm1(c: Test2_D):
    if False:
        return 10
    c.foo(_test_source())

def test2_noalarm1(c: Test2_C1):
    if False:
        print('Hello World!')
    c.foo(_test_source())

def test2_noalarm2(c: Test2_C2):
    if False:
        for i in range(10):
            print('nop')
    c.foo(_test_source())

def test2_noalarm3(c: Test2_C):
    if False:
        i = 10
        return i + 15
    c.foo(_test_source())

def misc_noalarm1(c: UnrelatedClass):
    if False:
        i = 10
        return i + 15
    _test_sink(c.attribute)

def misc_noalarm2(c: UnrelatedClass):
    if False:
        return 10
    _test_sink(c.instance)

def misc_noalarm3(c: UnrelatedClass):
    if False:
        print('Hello World!')
    c.foo(_test_source())