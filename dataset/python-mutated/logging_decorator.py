from typing import Awaitable, Callable

def with_logging_with_helper(f: Callable[[int], None]) -> Callable[[int], None]:
    if False:
        i = 10
        return i + 15

    def some_helper(x: int) -> None:
        if False:
            while True:
                i = 10
        print(x)
        eval(x)

    def inner(x: int) -> None:
        if False:
            i = 10
            return i + 15
        eval(x)
        f(x)
        some_helper(x)
    return inner

def with_logging_without_helper(f: Callable[[int], None]) -> Callable[[int], None]:
    if False:
        print('Hello World!')

    def inner(x: int) -> None:
        if False:
            return 10
        eval(x)
        f(x)
    return inner