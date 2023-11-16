from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, NewType, Optional, Type, TypeVar, Union, overload
from strawberry.exceptions import InvalidUnionTypeError
from strawberry.type import StrawberryType
from .utils.str_converters import to_camel_case
if TYPE_CHECKING:
    from graphql import GraphQLScalarType
if sys.version_info >= (3, 10):
    _T = TypeVar('_T', bound=Union[type, NewType])
else:
    _T = TypeVar('_T', bound=type)

def identity(x: _T) -> _T:
    if False:
        while True:
            i = 10
    return x

@dataclass
class ScalarDefinition(StrawberryType):
    name: str
    description: Optional[str]
    specified_by_url: Optional[str]
    serialize: Optional[Callable]
    parse_value: Optional[Callable]
    parse_literal: Optional[Callable]
    directives: Iterable[object] = ()
    implementation: Optional[GraphQLScalarType] = None
    _source_file: Optional[str] = None
    _source_line: Optional[int] = None

    def copy_with(self, type_var_map: Mapping[str, Union[StrawberryType, type]]) -> Union[StrawberryType, type]:
        if False:
            while True:
                i = 10
        return super().copy_with(type_var_map)

    @property
    def is_graphql_generic(self) -> bool:
        if False:
            print('Hello World!')
        return False

class ScalarWrapper:
    _scalar_definition: ScalarDefinition

    def __init__(self, wrap: Callable[[Any], Any]):
        if False:
            while True:
                i = 10
        self.wrap = wrap

    def __call__(self, *args: str, **kwargs: Any):
        if False:
            print('Hello World!')
        return self.wrap(*args, **kwargs)

    def __or__(self, other: Union[StrawberryType, type]) -> StrawberryType:
        if False:
            for i in range(10):
                print('nop')
        if other is None:
            return Optional[self]
        raise InvalidUnionTypeError(str(other), self.wrap)

def _process_scalar(cls: Type[_T], *, name: Optional[str]=None, description: Optional[str]=None, specified_by_url: Optional[str]=None, serialize: Optional[Callable]=None, parse_value: Optional[Callable]=None, parse_literal: Optional[Callable]=None, directives: Iterable[object]=()):
    if False:
        while True:
            i = 10
    from strawberry.exceptions.handler import should_use_rich_exceptions
    name = name or to_camel_case(cls.__name__)
    _source_file = None
    _source_line = None
    if should_use_rich_exceptions():
        frame = sys._getframe(3)
        _source_file = frame.f_code.co_filename
        _source_line = frame.f_lineno
    wrapper = ScalarWrapper(cls)
    wrapper._scalar_definition = ScalarDefinition(name=name, description=description, specified_by_url=specified_by_url, serialize=serialize, parse_literal=parse_literal, parse_value=parse_value, directives=directives, _source_file=_source_file, _source_line=_source_line)
    return wrapper

@overload
def scalar(*, name: Optional[str]=None, description: Optional[str]=None, specified_by_url: Optional[str]=None, serialize: Callable=identity, parse_value: Optional[Callable]=None, parse_literal: Optional[Callable]=None, directives: Iterable[object]=()) -> Callable[[_T], _T]:
    if False:
        return 10
    ...

@overload
def scalar(cls: _T, *, name: Optional[str]=None, description: Optional[str]=None, specified_by_url: Optional[str]=None, serialize: Callable=identity, parse_value: Optional[Callable]=None, parse_literal: Optional[Callable]=None, directives: Iterable[object]=()) -> _T:
    if False:
        i = 10
        return i + 15
    ...

def scalar(cls=None, *, name: Optional[str]=None, description: Optional[str]=None, specified_by_url: Optional[str]=None, serialize: Callable=identity, parse_value: Optional[Callable]=None, parse_literal: Optional[Callable]=None, directives: Iterable[object]=()) -> Any:
    if False:
        i = 10
        return i + 15
    'Annotates a class or type as a GraphQL custom scalar.\n\n    Example usages:\n\n    >>> strawberry.scalar(\n    >>>     datetime.date,\n    >>>     serialize=lambda value: value.isoformat(),\n    >>>     parse_value=datetime.parse_date\n    >>> )\n\n    >>> Base64Encoded = strawberry.scalar(\n    >>>     NewType("Base64Encoded", bytes),\n    >>>     serialize=base64.b64encode,\n    >>>     parse_value=base64.b64decode\n    >>> )\n\n    >>> @strawberry.scalar(\n    >>>     serialize=lambda value: ",".join(value.items),\n    >>>     parse_value=lambda value: CustomList(value.split(","))\n    >>> )\n    >>> class CustomList:\n    >>>     def __init__(self, items):\n    >>>         self.items = items\n\n    '
    if parse_value is None:
        parse_value = cls

    def wrap(cls: Type):
        if False:
            i = 10
            return i + 15
        return _process_scalar(cls, name=name, description=description, specified_by_url=specified_by_url, serialize=serialize, parse_value=parse_value, parse_literal=parse_literal, directives=directives)
    if cls is None:
        return wrap
    return wrap(cls)