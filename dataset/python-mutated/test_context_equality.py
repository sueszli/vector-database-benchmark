from typing import Callable
from returns.context import RequiresContext

def _same_function(some_arg: int) -> Callable[[float], float]:
    if False:
        print('Hello World!')
    return lambda other: other / some_arg

def test_equality():
    if False:
        return 10
    'Ensures that containers can be compared.'
    assert RequiresContext(_same_function) == RequiresContext(_same_function)

def test_nonequality():
    if False:
        while True:
            i = 10
    'Ensures that containers can be compared.'
    assert RequiresContext(_same_function) != RequiresContext(str)
    assert RequiresContext.from_value(1) != RequiresContext.from_value(1)