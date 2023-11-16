from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING, Optional
from .exception import StrawberryException
from .utils.source_finder import SourceFinder
if TYPE_CHECKING:
    from strawberry.types.fields.resolver import StrawberryResolver
    from .exception_source import ExceptionSource

class MissingReturnAnnotationError(StrawberryException):
    """The field is missing the return annotation"""

    def __init__(self, field_name: str, resolver: StrawberryResolver):
        if False:
            print('Hello World!')
        self.function = resolver.wrapped_func
        self.message = f'Return annotation missing for field "{field_name}", did you forget to add it?'
        self.rich_message = f'[bold red]Missing annotation for field `[underline]{resolver.name}[/]`'
        self.suggestion = f'To fix this error you can add an annotation, like so [italic]`def {resolver.name}(...) -> str:`'
        self.annotation_message = 'resolver missing annotation'

    @cached_property
    def exception_source(self) -> Optional[ExceptionSource]:
        if False:
            for i in range(10):
                print('nop')
        if self.function is None:
            return None
        source_finder = SourceFinder()
        return source_finder.find_function_from_object(self.function)