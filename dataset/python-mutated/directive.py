from __future__ import annotations
import dataclasses
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Generic, List, Optional, TypeVar
from typing_extensions import Annotated
from graphql import DirectiveLocation
from strawberry.field import StrawberryField
from strawberry.types.fields.resolver import INFO_PARAMSPEC, ReservedType, StrawberryResolver
from strawberry.unset import UNSET
if TYPE_CHECKING:
    import inspect
    from strawberry.arguments import StrawberryArgument

def directive_field(name: str, default: object=UNSET) -> Any:
    if False:
        for i in range(10):
            print('nop')
    return StrawberryField(python_name=None, graphql_name=name, default=default)
T = TypeVar('T')

class StrawberryDirectiveValue:
    ...
DirectiveValue = Annotated[T, StrawberryDirectiveValue()]
DirectiveValue.__doc__ = 'Represents the ``value`` argument for a GraphQL query directive.'
VALUE_PARAMSPEC = ReservedType(name='value', type=StrawberryDirectiveValue)

class StrawberryDirectiveResolver(StrawberryResolver[T]):
    RESERVED_PARAMSPEC = (INFO_PARAMSPEC, VALUE_PARAMSPEC)

    @cached_property
    def value_parameter(self) -> Optional[inspect.Parameter]:
        if False:
            for i in range(10):
                print('nop')
        return self.reserved_parameters.get(VALUE_PARAMSPEC)

@dataclasses.dataclass
class StrawberryDirective(Generic[T]):
    python_name: str
    graphql_name: Optional[str]
    resolver: StrawberryDirectiveResolver[T]
    locations: List[DirectiveLocation]
    description: Optional[str] = None

    @cached_property
    def arguments(self) -> List[StrawberryArgument]:
        if False:
            i = 10
            return i + 15
        return self.resolver.arguments

def directive(*, locations: List[DirectiveLocation], description: Optional[str]=None, name: Optional[str]=None) -> Callable[[Callable[..., T]], StrawberryDirective[T]]:
    if False:
        i = 10
        return i + 15

    def _wrap(f: Callable[..., T]) -> StrawberryDirective[T]:
        if False:
            while True:
                i = 10
        return StrawberryDirective(python_name=f.__name__, graphql_name=name, locations=locations, description=description, resolver=StrawberryDirectiveResolver(f))
    return _wrap
__all__ = ['DirectiveLocation', 'StrawberryDirective', 'directive']