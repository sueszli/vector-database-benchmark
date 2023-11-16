from builtins import _test_source

class C:

    def update(self, parameter):
        if False:
            for i in range(10):
                print('nop')
        ...

    def taint_parameter(self, tainted_parameter):
        if False:
            while True:
                i = 10
        ...

class D(C):

    def update(self, parameter):
        if False:
            return 10
        ...

    def taint_parameter(self, tainted_parameter):
        if False:
            print('Hello World!')
        ...

def test_obscure_tito():
    if False:
        print('Hello World!')
    c = C()
    c.update(_test_source())
    return c

def test_obscure_return():
    if False:
        i = 10
        return i + 15
    c = C()
    return c.update(_test_source())

def test_obscure_sink(parameter):
    if False:
        while True:
            i = 10
    c = C()
    c.taint_parameter(parameter)