from builtins import _test_sink, _test_source

class Wrapper:

    def __init__(self, a, b):
        if False:
            return 10
        self.a = a
        self.b = b

class C:

    def __init__(self, wrapper: Wrapper) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.x = wrapper
        self.y = wrapper.b

def y_is_benign():
    if False:
        for i in range(10):
            print('nop')
    wrapper = Wrapper(a=_test_source(), b=0)
    c = C(wrapper)
    _test_sink(c.y)