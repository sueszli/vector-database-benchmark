from typing import Any

def foo(a: int, *args: str, **kwargs: str) -> int:
    if False:
        return 10
    pass

def foo(a: Any, *args: str, **kwargs: str) -> int:
    if False:
        i = 10
        return i + 15
    pass

def foo(a: int, *args: str, **kwargs: str) -> Any:
    if False:
        i = 10
        return i + 15
    pass

def foo(a: int, *args: Any, **kwargs: Any) -> int:
    if False:
        return 10
    pass

def foo(a: int, *args: Any, **kwargs: str) -> int:
    if False:
        print('Hello World!')
    pass

def foo(a: int, *args: str, **kwargs: Any) -> int:
    if False:
        i = 10
        return i + 15
    pass

class Bar:

    def foo_method(self, a: int, *params: str, **options: str) -> int:
        if False:
            return 10
        pass

    def foo_method(self, a: Any, *params: str, **options: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        pass

    def foo_method(self, a: int, *params: str, **options: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        pass

    def foo_method(self, a: int, *params: Any, **options: Any) -> int:
        if False:
            for i in range(10):
                print('nop')
        pass

    def foo_method(self, a: int, *params: Any, **options: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        pass

    def foo_method(self, a: int, *params: str, **options: Any) -> int:
        if False:
            for i in range(10):
                print('nop')
        pass