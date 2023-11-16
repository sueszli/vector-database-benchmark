from builtins import _test_sink, _test_source

class StaticClass:

    @staticmethod
    def sink(oops):
        if False:
            while True:
                i = 10
        _test_sink(oops)

def test(source):
    if False:
        for i in range(10):
            print('nop')
    return StaticClass.sink(source)

def run_test(source):
    if False:
        print('Hello World!')
    test(_test_source())