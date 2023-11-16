from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING, Optional
from strawberry.exceptions.utils.source_finder import SourceFinder
from .exception import StrawberryException
if TYPE_CHECKING:
    from strawberry.field import StrawberryField
    from strawberry.object_type import StrawberryObjectDefinition
    from .exception_source import ExceptionSource

class UnresolvedFieldTypeError(StrawberryException):

    def __init__(self, type_definition: StrawberryObjectDefinition, field: StrawberryField):
        if False:
            while True:
                i = 10
        self.type_definition = type_definition
        self.field = field
        self.message = f"Could not resolve the type of '{self.field.name}'. Check that the class is accessible from the global module scope."
        self.rich_message = f"Could not resolve the type of [underline]'{self.field.name}'[/]. Check that the class is accessible from the global module scope."
        self.annotation_message = 'field defined here'
        self.suggestion = 'To fix this error you should either import the type or use LazyType.'
        super().__init__(self.message)

    @cached_property
    def exception_source(self) -> Optional[ExceptionSource]:
        if False:
            for i in range(10):
                print('nop')
        source_finder = SourceFinder()
        source = source_finder.find_class_attribute_from_object(self.type_definition.origin, self.field.name)
        if source is not None:
            return source
        if self.field.base_resolver:
            return source_finder.find_function_from_object(self.field.base_resolver.wrapped_func)
        return None