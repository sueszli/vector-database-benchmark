from typing import TypeVar
from typing_extensions import Annotated
from strawberry.utils.typing import type_has_annotation

class StrawberryPrivate:
    ...
T = TypeVar('T')
Private = Annotated[T, StrawberryPrivate()]
Private.__doc__ = "Represents a field that won't be exposed in the GraphQL schema\n\nExample:\n\n>>> import strawberry\n>>> @strawberry.type\n... class User:\n...     name: str\n...     age: strawberry.Private[int]\n"

def is_private(type_: object) -> bool:
    if False:
        print('Hello World!')
    return type_has_annotation(type_, StrawberryPrivate)