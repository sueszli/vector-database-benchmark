from __future__ import annotations
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Optional
from .exception import StrawberryException
from .utils.source_finder import SourceFinder
if TYPE_CHECKING:
    from .exception_source import ExceptionSource

class ObjectIsNotClassError(StrawberryException):

    class MethodType(Enum):
        INPUT = 'input'
        INTERFACE = 'interface'
        TYPE = 'type'

    def __init__(self, obj: object, method_type: MethodType):
        if False:
            i = 10
            return i + 15
        self.obj = obj
        self.function = obj
        obj_name = obj.__name__
        self.message = f'strawberry.{method_type.value} can only be used with class types. Provided object {obj_name} is not a type.'
        self.rich_message = f'strawberry.{method_type.value} can only be used with class types. Provided object `[underline]{obj_name}[/]` is not a type.'
        self.annotation_message = 'function defined here'
        self.suggestion = f'To fix this error, make sure your use strawberry.{method_type.value} on a class.'
        super().__init__(self.message)

    @classmethod
    def input(cls, obj: object) -> ObjectIsNotClassError:
        if False:
            return 10
        return cls(obj, cls.MethodType.INPUT)

    @classmethod
    def interface(cls, obj: object) -> ObjectIsNotClassError:
        if False:
            while True:
                i = 10
        return cls(obj, cls.MethodType.INTERFACE)

    @classmethod
    def type(cls, obj: object) -> ObjectIsNotClassError:
        if False:
            for i in range(10):
                print('nop')
        return cls(obj, cls.MethodType.TYPE)

    @cached_property
    def exception_source(self) -> Optional[ExceptionSource]:
        if False:
            for i in range(10):
                print('nop')
        if self.function is None:
            return None
        source_finder = SourceFinder()
        return source_finder.find_function_from_object(self.function)