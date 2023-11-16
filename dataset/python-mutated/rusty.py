"""An error-handling model influenced by that used by the Rust programming language

See https://doc.rust-lang.org/book/ch09-00-error-handling.html.
"""
from typing import Generic, TypeVar, Union
T = TypeVar('T')
E = TypeVar('E', bound=Exception)

class Ok(Generic[T]):

    def __init__(self, value: T) -> None:
        if False:
            print('Hello World!')
        self._value = value

    def ok(self) -> T:
        if False:
            while True:
                i = 10
        return self._value

class Err(Generic[E]):

    def __init__(self, e: E) -> None:
        if False:
            while True:
                i = 10
        self._e = e

    def err(self) -> E:
        if False:
            print('Hello World!')
        return self._e
Result = Union[Ok[T], Err[E]]