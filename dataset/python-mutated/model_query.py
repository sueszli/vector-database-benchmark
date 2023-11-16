from builtins import _test_sink, _test_source
from functools import lru_cache

def foo(x):
    if False:
        for i in range(10):
            print('nop')
    return 0

def barfoo(x):
    if False:
        print('Hello World!')
    return 0

class C:

    def foo(self, x):
        if False:
            i = 10
            return i + 15
        return 0

    def positional_method(self, x):
        if False:
            print('Hello World!')
        return 0

def two_parameters(x, y):
    if False:
        print('Hello World!')
    return 0

def three_parameters(x, y, z):
    if False:
        while True:
            i = 10
    return 0

def positional_a(x):
    if False:
        return 10
    return 0

def positional_b(y):
    if False:
        for i in range(10):
            print('nop')
    return 0

class Base:

    def foo(self, x):
        if False:
            i = 10
            return i + 15
        return 0

class NotBase:

    def foo(self, x):
        if False:
            while True:
                i = 10
        return 0

class Child(Base):

    def bar(self, y):
        if False:
            return 10
        return 0

class GrandChild(Child):

    def baz(self, z):
        if False:
            i = 10
            return i + 15
        return 0

@lru_cache(maxsize=1)
def positional_decorated(x, y) -> int:
    if False:
        print('Hello World!')
    ...

class AttributeTestBase:
    ...

class AttributeTestClass1(AttributeTestBase):
    attribute: ...

    def __init__(self):
        if False:
            print('Hello World!')
        self.instance = None

class AttributeTestClass2(AttributeTestBase):
    attribute: ...

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.instance = None

class AttributeTestClass3:
    attribute: ...

    def __init__(self):
        if False:
            print('Hello World!')
        self.instance = None

class AttributeTestClass4:
    attribute: ...

    def __init__(self):
        if False:
            print('Hello World!')
        self.instance = None

class AttributeTestClass5:
    foo_attribute: ...

    def __init__(self):
        if False:
            print('Hello World!')
        self.foo_instance = None

class AttributeTestClass6:
    foo_attribute: ...

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.foo_instance = None

class AttributeTestClass7:
    nomatch_attribute1: ...

    def __init__(self):
        if False:
            while True:
                i = 10
        self.nomatch_instance1 = None

class AttributeTestClass8(AttributeTestClass7):
    nomatch_attribute2: ...

    def __init__(self):
        if False:
            while True:
                i = 10
        self.nomatch_instance2 = None

def alarm_1(x: AttributeTestClass1):
    if False:
        while True:
            i = 10
    _test_sink(x.attribute)

def alarm_2(x: AttributeTestClass1):
    if False:
        for i in range(10):
            print('nop')
    _test_sink(x.instance)

def alarm_3(x: AttributeTestClass2):
    if False:
        return 10
    _test_sink(x.attribute)

def alarm_4(x: AttributeTestClass2):
    if False:
        i = 10
        return i + 15
    _test_sink(x.instance)

def alarm_5(x: AttributeTestClass3, source):
    if False:
        for i in range(10):
            print('nop')
    x.attribute = source

def alarm_6(x: AttributeTestClass3):
    if False:
        return 10
    x.instance = _test_source()

def alarm_7(x: AttributeTestClass4):
    if False:
        return 10
    return x.attribute

def alarm_8(x: AttributeTestClass4):
    if False:
        return 10
    _test_sink(x.instance)

def alarm_9(x: AttributeTestClass5):
    if False:
        while True:
            i = 10
    _test_sink(x.foo_attribute)

def alarm_10(x: AttributeTestClass5):
    if False:
        for i in range(10):
            print('nop')
    _test_sink(x.foo_instance)

def alarm_11(x: AttributeTestClass6):
    if False:
        return 10
    _test_sink(x.foo_attribute)

def alarm_12(x: AttributeTestClass6):
    if False:
        while True:
            i = 10
    _test_sink(x.foo_instance)

def no_alarm_1(x: AttributeTestClass7):
    if False:
        print('Hello World!')
    _test_sink(x.nomatch_attribute1)
    _test_sink(x.nomatch_instance1)

def no_alarm_2(x: AttributeTestClass8):
    if False:
        i = 10
        return i + 15
    _test_sink(x.nomatch_instance2)
    _test_sink(x.nomatch_instance2)

def function_test1_alarm1():
    if False:
        while True:
            i = 10
    return 0

def function_test1_alarm2():
    if False:
        print('Hello World!')
    return 0

def function_test1_noalarm1():
    if False:
        i = 10
        return i + 15
    return 0

def function_test1_noalarm2():
    if False:
        while True:
            i = 10
    return 0

class ClassTest1:

    def method_test1_alarm1():
        if False:
            while True:
                i = 10
        return 0

    def method_test1_noalarm1():
        if False:
            i = 10
            return i + 15
        return 0

class NoAlarmClass:
    ...

class ClassTest2_Alarm1:

    def method1():
        if False:
            return 10
        return 0

    def method2():
        if False:
            return 10
        return 0

class ClassTest2_NoAlarm1(NoAlarmClass):

    def method1():
        if False:
            while True:
                i = 10
        return 0

    def method2():
        if False:
            for i in range(10):
                print('nop')
        return 0

class ClassTest3_Alarm1:

    def method1():
        if False:
            for i in range(10):
                print('nop')
        return 0

    def method2():
        if False:
            print('Hello World!')
        return 0