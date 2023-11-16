from builtins import _test_sink, _test_source

class Object:
    pass
obj1 = Object()

def obj1_source():
    if False:
        return 10
    obj1.x = _test_source()

def obj1_sink():
    if False:
        print('Hello World!')
    _test_sink(obj1.x)

def obj1_flow():
    if False:
        return 10
    obj1_source()
    obj1_sink()

def obj1_no_flow():
    if False:
        print('Hello World!')
    obj1_sink()
    obj1_source()
obj2 = Object()

def obj2_sink():
    if False:
        while True:
            i = 10
    _test_sink(obj2)
obj3 = Object()

def obj3_return():
    if False:
        i = 10
        return i + 15
    return obj3

def obj3_set(x):
    if False:
        for i in range(10):
            print('nop')
    obj3.x = x

def obj3_flow():
    if False:
        return 10
    obj3_set(_test_source())
    y = obj3_return()
    _test_sink(y.x)
obj4 = _test_source()

def obj4_flow():
    if False:
        print('Hello World!')
    _test_sink(obj4)

def create_global_source():
    if False:
        print('Hello World!')
    global z
    z = _test_source()
create_global_source()

def return_global_source():
    if False:
        for i in range(10):
            print('nop')
    return z
obj6 = Object()

def obj6_source():
    if False:
        for i in range(10):
            print('nop')
    global obj6
    obj6 = _test_source()

def obj6_sink():
    if False:
        while True:
            i = 10
    _test_sink(obj6)

def obj6_flow():
    if False:
        for i in range(10):
            print('nop')
    obj6_source()
    obj6_sink()

def obj7_source():
    if False:
        return 10
    global obj7
    obj7 = _test_source()

def obj7_sink():
    if False:
        for i in range(10):
            print('nop')
    _test_sink(obj7)

def obj7_flow():
    if False:
        for i in range(10):
            print('nop')
    obj7_source()
    obj7_sink()
obj8 = Object()

def obj8_return():
    if False:
        i = 10
        return i + 15
    return obj8

def obj8_set(x):
    if False:
        print('Hello World!')
    global obj8
    obj8 = x

def obj8_flow():
    if False:
        while True:
            i = 10
    obj8_set(_test_source())
    y = obj8_return()
    _test_sink(y)