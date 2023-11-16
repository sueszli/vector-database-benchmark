from typing import Any
import typing

class Bad:

    def __eq__(self, other: Any) -> bool:
        if False:
            i = 10
            return i + 15
        ...

    def __ne__(self, other: typing.Any) -> typing.Any:
        if False:
            i = 10
            return i + 15
        ...

class Good:

    def __eq__(self, other: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __ne__(self, obj: object) -> int:
        if False:
            print('Hello World!')
        ...

class WeirdButFine:

    def __eq__(self, other: Any, strange_extra_arg: list[str]) -> Any:
        if False:
            for i in range(10):
                print('nop')
        ...

    def __ne__(self, *, kw_only_other: Any) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ...

class Unannotated:

    def __eq__(self) -> Any:
        if False:
            while True:
                i = 10
        ...

    def __ne__(self) -> bool:
        if False:
            while True:
                i = 10
        ...