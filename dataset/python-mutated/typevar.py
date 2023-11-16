from builtins import _test_sink
from typing import TypeVar
T = TypeVar('T', bound='Foo')

class Foo:

    def __init__(self, tainted: str) -> None:
        if False:
            return 10
        self.tainted: str = tainted

def issue(foo: T) -> T:
    if False:
        for i in range(10):
            print('nop')
    _test_sink(foo.tainted)
    return foo