""" Internal primitives of the properties system. """
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, TypeVar, Union
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
__all__ = ('Intrinsic', 'Optional', 'Undefined')
T = TypeVar('T')

class UndefinedType:
    """ Indicates no value set, which is not the same as setting ``None``. """

    def __copy__(self) -> UndefinedType:
        if False:
            i = 10
            return i + 15
        return self

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return 'Undefined'

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'Undefined'
Undefined = UndefinedType()
Optional: TypeAlias = Union[T, UndefinedType]

class IntrinsicType:
    """ Indicates usage of the intrinsic default value of a property. """

    def __copy__(self) -> IntrinsicType:
        if False:
            print('Hello World!')
        return self

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return 'Intrinsic'

    def __repr__(self) -> str:
        if False:
            return 10
        return 'Intrinsic'
Intrinsic = IntrinsicType()