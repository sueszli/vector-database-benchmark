from builtins import _test_sink, _test_source

class A:
    a = ''
    b = ''

    def __init__(self, c):
        if False:
            print('Hello World!')
        A.b = _test_source()
        self.c = c
        self.d = _test_source()

    def sink_a(self):
        if False:
            i = 10
            return i + 15
        _test_sink(A.a)

    def sink_b(self):
        if False:
            print('Hello World!')
        _test_sink(A.b)

    def sink_c(self):
        if False:
            for i in range(10):
                print('nop')
        _test_sink(self.c)

    def sink_d(self):
        if False:
            while True:
                i = 10
        _test_sink(self.d)

def class_attribute_A_a_source():
    if False:
        for i in range(10):
            print('nop')
    A.a = _test_source()

def class_attribute_A_a_sink():
    if False:
        i = 10
        return i + 15
    _test_sink(A.a)

def class_attribute_A_a_flow():
    if False:
        i = 10
        return i + 15
    class_attribute_A_a_source()
    class_attribute_A_a_sink()

def class_attribute_A_a_no_flow():
    if False:
        return 10
    class_attribute_A_a_sink()
    class_attribute_A_a_source()

def class_attribute_A_b_sink():
    if False:
        i = 10
        return i + 15
    _test_sink(A.b)

def class_attribute_A_b_flow1():
    if False:
        print('Hello World!')
    A()
    class_attribute_A_b_sink()

def class_attribute_A_b_flow2():
    if False:
        for i in range(10):
            print('nop')
    A().sink_b()

def instance_attribute_A_c_no_flow():
    if False:
        print('Hello World!')
    A().sink_c()

def instance_attribute_A_d_flow():
    if False:
        return 10
    A().sink_d()