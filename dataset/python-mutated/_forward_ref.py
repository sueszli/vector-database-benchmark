from __future__ import annotations as _annotations
from dataclasses import dataclass
from typing import Union

@dataclass
class PydanticRecursiveRef:
    type_ref: str
    __name__ = 'PydanticRecursiveRef'
    __hash__ = object.__hash__

    def __call__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Defining __call__ is necessary for the `typing` module to let you use an instance of\n        this class as the result of resolving a standard ForwardRef.\n        '

    def __or__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return Union[self, other]

    def __ror__(self, other):
        if False:
            print('Hello World!')
        return Union[other, self]