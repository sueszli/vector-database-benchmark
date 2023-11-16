from builtins import _test_sink, _test_source

def test():
    if False:
        i = 10
        return i + 15

    class A:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            return _test_sink(_test_source())
    A()