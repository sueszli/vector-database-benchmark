from builtins import _test_sink, _test_source

class MyBaseClass:

    def __init__(self, argument: str) -> None:
        if False:
            print('Hello World!')
        _test_sink(argument)

class MyDerivedClass(MyBaseClass):
    variable = ''

def test() -> None:
    if False:
        for i in range(10):
            print('nop')
    derived = MyDerivedClass(_test_source())