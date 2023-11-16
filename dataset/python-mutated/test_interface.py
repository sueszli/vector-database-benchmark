from dataclasses import dataclass
from typing import Any, List
import pytest
from pytest_mock import MockerFixture
import strawberry
from strawberry.types.types import StrawberryObjectDefinition

def test_query_interface():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.interface
    class Cheese:
        name: str

    @strawberry.type
    class Swiss(Cheese):
        canton: str

    @strawberry.type
    class Italian(Cheese):
        province: str

    @strawberry.type
    class Root:

        @strawberry.field
        def assortment(self) -> List[Cheese]:
            if False:
                print('Hello World!')
            return [Italian(name='Asiago', province='Friuli'), Swiss(name='Tomme', canton='Vaud')]
    schema = strawberry.Schema(query=Root, types=[Swiss, Italian])
    query = '{\n        assortment {\n            name\n            ... on Italian { province }\n            ... on Swiss { canton }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data is not None
    assert result.data['assortment'] == [{'name': 'Asiago', 'province': 'Friuli'}, {'canton': 'Vaud', 'name': 'Tomme'}]

def test_interfaces_can_implement_other_interfaces():
    if False:
        while True:
            i = 10

    @strawberry.interface
    class Error:
        message: str

    @strawberry.interface
    class FieldError(Error):
        message: str
        field: str

    @strawberry.type
    class PasswordTooShort(FieldError):
        message: str
        field: str
        fix: str

    @strawberry.type
    class Query:

        @strawberry.field
        def always_error(self) -> Error:
            if False:
                print('Hello World!')
            return PasswordTooShort(message='Password Too Short', field='Password', fix='Choose more characters')
    schema = strawberry.Schema(Query, types=[PasswordTooShort])
    query = '{\n        alwaysError {\n            ... on Error {\n                message\n            }\n            ... on FieldError {\n                field\n            }\n            ... on PasswordTooShort {\n                fix\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data is not None
    assert result.data['alwaysError'] == {'message': 'Password Too Short', 'field': 'Password', 'fix': 'Choose more characters'}

def test_interface_duck_typing():
    if False:
        print('Hello World!')

    @strawberry.interface
    class Entity:
        id: int

    @strawberry.type
    class Anime(Entity):
        name: str

        @classmethod
        def is_type_of(cls, obj: Any, _) -> bool:
            if False:
                print('Hello World!')
            return isinstance(obj, AnimeORM)

    @dataclass
    class AnimeORM:
        id: int
        name: str

    @strawberry.type
    class Query:

        @strawberry.field
        def anime(self) -> Entity:
            if False:
                while True:
                    i = 10
            return AnimeORM(id=1, name='One Piece')
    schema = strawberry.Schema(query=Query, types=[Anime])
    query = '{\n        anime { id ... on Anime { name } }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'anime': {'id': 1, 'name': 'One Piece'}}

def test_interface_explicit_type_resolution():
    if False:
        while True:
            i = 10

    @dataclass
    class AnimeORM:
        id: int
        name: str

    @strawberry.interface
    class Node:
        id: int

    @strawberry.type
    class Anime(Node):
        name: str

        @classmethod
        def is_type_of(cls, obj: Any, _) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return isinstance(obj, AnimeORM)

    @strawberry.type
    class Query:

        @strawberry.field
        def node(self) -> Node:
            if False:
                for i in range(10):
                    print('nop')
            return AnimeORM(id=1, name='One Piece')
    schema = strawberry.Schema(query=Query, types=[Anime])
    query = '{ node { __typename, id ... on Anime { name }} }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'node': {'__typename': 'Anime', 'id': 1, 'name': 'One Piece'}}

@pytest.mark.xfail(reason="We don't support returning dictionaries yet")
def test_interface_duck_typing_returning_dict():
    if False:
        i = 10
        return i + 15

    @strawberry.interface
    class Entity:
        id: int

    @strawberry.type
    class Anime(Entity):
        name: str

    @strawberry.type
    class Query:

        @strawberry.field
        def anime(self) -> Anime:
            if False:
                print('Hello World!')
            return dict(id=1, name='One Piece')
    schema = strawberry.Schema(query=Query)
    query = '{\n        anime { name }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'anime': {'name': 'One Piece'}}

def test_duplicated_interface_in_multi_inheritance():
    if False:
        for i in range(10):
            print('nop')
    'Test that interfaces are gathered properly via CPython\'s MRO.\n\n    Previously interfaces were duplicated within a "Diamond Problem" inheritance\n    scenario which is tested here. Using the MRO instead of the `__bases__` attribute of\n    a class in :py:func:`strawberry.object_type._get_interfaces` allows Python\'s C3\n    linearization algorithm to create a consistent precedents graph without duplicates.\n    '

    @strawberry.interface
    class Base:
        id: str

    @strawberry.interface
    class InterfaceA(Base):
        id: str
        field_a: str

    @strawberry.interface
    class InterfaceB(Base):
        id: str
        field_b: str

    @strawberry.type
    class MyType(InterfaceA, InterfaceB):
        id: str
        field_a: str
        field_b: str

    @strawberry.type
    class Query:
        my_type: MyType
    type_definition: StrawberryObjectDefinition = MyType.__strawberry_definition__
    origins = [i.origin for i in type_definition.interfaces]
    assert origins == [InterfaceA, InterfaceB, Base]
    strawberry.Schema(Query)

def test_interface_resolve_type(mocker: MockerFixture):
    if False:
        while True:
            i = 10
    'Check that the default implemenetation of `resolve_type` functions as expected.\n\n    In this test-case the default implementation of `resolve_type` defined in\n    `GraphQLCoreConverter.from_interface`, should immediately resolve the type of the\n    returned concrete object. A concrete object is defined as one that is an instance of\n    the interface it implements.\n\n    Before the default implementation of `resolve_type`, the `is_type_of` methods of all\n    specializations of an interface (in this case Anime & Movie) would be called. As\n    this needlessly reduces performance, this test checks if only `Anime.is_type_of` is\n    called when `Query.node` returns an `Anime` object.\n    '

    class IsTypeOfTester:

        @classmethod
        def is_type_of(cls, obj: Any, _) -> bool:
            if False:
                while True:
                    i = 10
            return isinstance(obj, cls)
    spy_is_type_of = mocker.spy(IsTypeOfTester, 'is_type_of')

    @strawberry.interface
    class Node:
        id: int

    @strawberry.type
    class Anime(Node, IsTypeOfTester):
        name: str

    @strawberry.type
    class Movie(Node):
        title: str

        @classmethod
        def is_type_of(cls, *args: Any, **kwargs: Any) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            del args, kwargs
            raise RuntimeError("Movie.is_type_of shouldn't have been called")

    @strawberry.type
    class Query:

        @strawberry.field
        def node(self) -> Node:
            if False:
                i = 10
                return i + 15
            return Anime(id=1, name='One Pierce')
    schema = strawberry.Schema(query=Query, types=[Anime, Movie])
    query = '{ node {  __typename, id } }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'node': {'__typename': 'Anime', 'id': 1}}
    spy_is_type_of.assert_called_once()

def test_interface_specialized_resolve_type(mocker: MockerFixture):
    if False:
        print('Hello World!')
    'Test that a specialized ``resolve_type`` is called.'

    class InterfaceTester:

        @classmethod
        def resolve_type(cls, obj: Any, *args: Any, **kwargs: Any) -> str:
            if False:
                print('Hello World!')
            del args, kwargs
            return obj.__strawberry_definition__.name
    spy_resolve_type = mocker.spy(InterfaceTester, 'resolve_type')

    @strawberry.interface
    class Food(InterfaceTester):
        id: int

    @strawberry.type
    class Fruit(Food):
        name: str

    @strawberry.type
    class Query:

        @strawberry.field
        def food(self) -> Food:
            if False:
                print('Hello World!')
            return Fruit(id=1, name='strawberry')
    schema = strawberry.Schema(query=Query, types=[Fruit])
    result = schema.execute_sync('query { food { ... on Fruit { name } } }')
    assert not result.errors
    assert result.data == {'food': {'name': 'strawberry'}}
    spy_resolve_type.assert_called_once()

@pytest.mark.asyncio
async def test_derived_interface(mocker: MockerFixture):
    """Test if correct resolve_type is called on a derived interface."""

    class NodeInterfaceTester:

        @classmethod
        def resolve_type(cls, obj: Any, *args: Any, **kwargs: Any) -> str:
            if False:
                while True:
                    i = 10
            del args, kwargs
            return obj.__strawberry_definition__.name

    class NamedNodeInterfaceTester:

        @classmethod
        def resolve_type(cls, obj: Any, *args: Any, **kwargs: Any) -> str:
            if False:
                for i in range(10):
                    print('nop')
            del args, kwargs
            return obj.__strawberry_definition__.name
    spy_node_resolve_type = mocker.spy(NodeInterfaceTester, 'resolve_type')
    spy_named_node_resolve_type = mocker.spy(NamedNodeInterfaceTester, 'resolve_type')

    @strawberry.interface
    class Node(NodeInterfaceTester):
        id: int

    @strawberry.interface
    class NamedNode(NamedNodeInterfaceTester, Node):
        name: str

    @strawberry.type
    class Person(NamedNode):
        pass

    @strawberry.type
    class Query:

        @strawberry.field
        def friends(self) -> List[NamedNode]:
            if False:
                for i in range(10):
                    print('nop')
            return [Person(id=1, name='foo'), Person(id=2, name='bar')]
    schema = strawberry.Schema(Query, types=[Person])
    result = await schema.execute('query { friends { name } }')
    assert not result.errors
    assert result.data == {'friends': [{'name': 'foo'}, {'name': 'bar'}]}
    assert result.data is not None
    assert spy_named_node_resolve_type.call_count == len(result.data['friends'])
    spy_node_resolve_type.assert_not_called()

def test_resolve_type_on_interface_returning_interface():
    if False:
        while True:
            i = 10

    @strawberry.interface
    class Node:
        id: strawberry.ID

        @classmethod
        def resolve_type(cls, obj: Any, *args: Any, **kwargs: Any) -> str:
            if False:
                while True:
                    i = 10
            return 'Video' if obj.id == '1' else 'Image'

    @strawberry.type
    class Video(Node):
        ...

    @strawberry.type
    class Image(Node):
        ...

    @strawberry.type
    class Query:

        @strawberry.field
        def node(self, id: strawberry.ID) -> Node:
            if False:
                print('Hello World!')
            return Node(id=id)
    schema = strawberry.Schema(query=Query, types=[Video, Image])
    query = '\n        query {\n            one: node(id: "1") {\n                __typename\n                id\n            }\n            two: node(id: "2") {\n                __typename\n                id\n            }\n        }\n    '
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data
    assert result.data['one'] == {'id': '1', '__typename': 'Video'}
    assert result.data['two'] == {'id': '2', '__typename': 'Image'}