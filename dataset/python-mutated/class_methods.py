from builtins import _test_sink, _test_source

class Test:

    @classmethod
    def foo(cls, x) -> None:
        if False:
            while True:
                i = 10
        return _test_sink(x)

def bar():
    if False:
        for i in range(10):
            print('nop')
    Test.foo(_test_source())