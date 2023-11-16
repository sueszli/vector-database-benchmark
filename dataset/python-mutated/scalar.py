import sys
from typing import Any, Callable, Iterable, NewType, Optional, Type, TypeVar, Union, overload
from strawberry.custom_scalar import _process_scalar
if sys.version_info >= (3, 10):
    _T = TypeVar('_T', bound=Union[type, NewType])
else:
    _T = TypeVar('_T', bound=type)

def identity(x: _T) -> _T:
    if False:
        print('Hello World!')
    return x

@overload
def scalar(*, name: Optional[str]=None, description: Optional[str]=None, specified_by_url: Optional[str]=None, serialize: Callable=identity, parse_value: Optional[Callable]=None, parse_literal: Optional[Callable]=None, directives: Iterable[object]=(), inaccessible: bool=False, tags: Optional[Iterable[str]]=()) -> Callable[[_T], _T]:
    if False:
        i = 10
        return i + 15
    ...

@overload
def scalar(cls: _T, *, name: Optional[str]=None, description: Optional[str]=None, specified_by_url: Optional[str]=None, serialize: Callable=identity, parse_value: Optional[Callable]=None, parse_literal: Optional[Callable]=None, directives: Iterable[object]=(), inaccessible: bool=False, tags: Optional[Iterable[str]]=()) -> _T:
    if False:
        for i in range(10):
            print('nop')
    ...

def scalar(cls=None, *, name: Optional[str]=None, description: Optional[str]=None, specified_by_url: Optional[str]=None, serialize: Callable=identity, parse_value: Optional[Callable]=None, parse_literal: Optional[Callable]=None, directives: Iterable[object]=(), inaccessible: bool=False, tags: Optional[Iterable[str]]=()) -> Any:
    if False:
        while True:
            i = 10
    'Annotates a class or type as a GraphQL custom scalar.\n\n    Example usages:\n\n    >>> strawberry.federation.scalar(\n    >>>     datetime.date,\n    >>>     serialize=lambda value: value.isoformat(),\n    >>>     parse_value=datetime.parse_date\n    >>> )\n\n    >>> Base64Encoded = strawberry.federation.scalar(\n    >>>     NewType("Base64Encoded", bytes),\n    >>>     serialize=base64.b64encode,\n    >>>     parse_value=base64.b64decode\n    >>> )\n\n    >>> @strawberry.federation.scalar(\n    >>>     serialize=lambda value: ",".join(value.items),\n    >>>     parse_value=lambda value: CustomList(value.split(","))\n    >>> )\n    >>> class CustomList:\n    >>>     def __init__(self, items):\n    >>>         self.items = items\n\n    '
    from strawberry.federation.schema_directives import Inaccessible, Tag
    if parse_value is None:
        parse_value = cls
    directives = list(directives)
    if inaccessible:
        directives.append(Inaccessible())
    if tags:
        directives.extend((Tag(name=tag) for tag in tags))

    def wrap(cls: Type):
        if False:
            print('Hello World!')
        return _process_scalar(cls, name=name, description=description, specified_by_url=specified_by_url, serialize=serialize, parse_value=parse_value, parse_literal=parse_literal, directives=directives)
    if cls is None:
        return wrap
    return wrap(cls)