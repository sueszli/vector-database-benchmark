import sys
from typing import Generic, TypeVar, Union
from typing_extensions import Annotated
import pytest
import strawberry
from strawberry.annotation import StrawberryAnnotation
from strawberry.exceptions import InvalidUnionTypeError
from strawberry.type import get_object_definition
from strawberry.union import StrawberryUnion, union

def test_python_union():
    if False:
        while True:
            i = 10

    @strawberry.type
    class User:
        name: str

    @strawberry.type
    class Error:
        name: str
    annotation = StrawberryAnnotation(Union[User, Error])
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryUnion)
    assert resolved.types == (User, Error)
    assert resolved == StrawberryUnion(name=None, type_annotations=(StrawberryAnnotation(User), StrawberryAnnotation(Error)))
    assert resolved == Union[User, Error]

@pytest.mark.skipif(sys.version_info < (3, 10), reason='short syntax for union is only available on python 3.10+')
def test_python_union_short_syntax():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class User:
        name: str

    @strawberry.type
    class Error:
        name: str
    annotation = StrawberryAnnotation(User | Error)
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryUnion)
    assert resolved.types == (User, Error)
    assert resolved == StrawberryUnion(name=None, type_annotations=(StrawberryAnnotation(User), StrawberryAnnotation(Error)))
    assert resolved == Union[User, Error]

def test_strawberry_union():
    if False:
        print('Hello World!')

    @strawberry.type
    class User:
        name: str

    @strawberry.type
    class Error:
        name: str
    cool_union = Annotated[Union[User, Error], union(name='CoolUnion')]
    annotation = StrawberryAnnotation(cool_union)
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryUnion)
    assert resolved.types == (User, Error)
    assert resolved == StrawberryUnion(name='CoolUnion', type_annotations=(StrawberryAnnotation(User), StrawberryAnnotation(Error)))

def test_union_with_generic():
    if False:
        return 10
    T = TypeVar('T')

    @strawberry.type
    class Error:
        message: str

    @strawberry.type
    class Edge(Generic[T]):
        node: T
    Result = Annotated[Union[Error, Edge[str]], strawberry.union('Result')]
    strawberry_union = StrawberryAnnotation(Result).resolve()
    assert isinstance(strawberry_union, StrawberryUnion)
    assert strawberry_union.graphql_name == 'Result'
    assert strawberry_union.types[0] == Error
    assert get_object_definition(strawberry_union.types[1], strict=True).is_graphql_generic is False

@pytest.mark.raises_strawberry_exception(InvalidUnionTypeError, match='Type `int` cannot be used in a GraphQL Union')
def test_error_with_scalar_types():
    if False:
        print('Hello World!')
    Something = Annotated[Union[int, str, float, bool], strawberry.union('Something')]

    @strawberry.type
    class Query:
        something: Something
    schema = strawberry.Schema(query=Query)

@pytest.mark.raises_strawberry_exception(InvalidUnionTypeError, match='Type `int` cannot be used in a GraphQL Union')
@pytest.mark.skipif(sys.version_info < (3, 10), reason='short syntax for union is only available on python 3.10+')
def test_error_with_scalar_types_pipe():
    if False:
        return 10
    Something2 = Annotated[int | str | float | bool, strawberry.union('Something2')]

    @strawberry.type
    class Query:
        something: Something2
    schema = strawberry.Schema(query=Query)