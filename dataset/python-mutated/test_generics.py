import textwrap
from enum import Enum
from typing import Any, Generic, List, Optional, TypeVar, Union
from typing_extensions import Self
import pytest
import strawberry

def test_supports_generic_simple_type():
    if False:
        while True:
            i = 10
    T = TypeVar('T')

    @strawberry.type
    class Edge(Generic[T]):
        cursor: strawberry.ID
        node_field: T

    @strawberry.type
    class Query:

        @strawberry.field
        def example(self) -> Edge[int]:
            if False:
                i = 10
                return i + 15
            return Edge(cursor=strawberry.ID('1'), node_field=1)
    schema = strawberry.Schema(query=Query)
    query = '{\n        example {\n            __typename\n            cursor\n            nodeField\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'example': {'__typename': 'IntEdge', 'cursor': '1', 'nodeField': 1}}

def test_supports_generic_specialized():
    if False:
        i = 10
        return i + 15
    T = TypeVar('T')

    @strawberry.type
    class Edge(Generic[T]):
        cursor: strawberry.ID
        node_field: T

    @strawberry.type
    class IntEdge(Edge[int]):
        ...

    @strawberry.type
    class Query:

        @strawberry.field
        def example(self) -> IntEdge:
            if False:
                i = 10
                return i + 15
            return IntEdge(cursor=strawberry.ID('1'), node_field=1)
    schema = strawberry.Schema(query=Query)
    query = '{\n        example {\n            __typename\n            cursor\n            nodeField\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'example': {'__typename': 'IntEdge', 'cursor': '1', 'nodeField': 1}}

def test_supports_generic_specialized_subclass():
    if False:
        return 10
    T = TypeVar('T')

    @strawberry.type
    class Edge(Generic[T]):
        cursor: strawberry.ID
        node_field: T

    @strawberry.type
    class IntEdge(Edge[int]):
        ...

    @strawberry.type
    class IntEdgeSubclass(IntEdge):
        ...

    @strawberry.type
    class Query:

        @strawberry.field
        def example(self) -> IntEdgeSubclass:
            if False:
                i = 10
                return i + 15
            return IntEdgeSubclass(cursor=strawberry.ID('1'), node_field=1)
    schema = strawberry.Schema(query=Query)
    query = '{\n        example {\n            __typename\n            cursor\n            nodeField\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'example': {'__typename': 'IntEdgeSubclass', 'cursor': '1', 'nodeField': 1}}

def test_supports_generic_specialized_with_type():
    if False:
        while True:
            i = 10
    T = TypeVar('T')

    @strawberry.type
    class Fruit:
        name: str

    @strawberry.type
    class Edge(Generic[T]):
        cursor: strawberry.ID
        node_field: T

    @strawberry.type
    class FruitEdge(Edge[Fruit]):
        ...

    @strawberry.type
    class Query:

        @strawberry.field
        def example(self) -> FruitEdge:
            if False:
                for i in range(10):
                    print('nop')
            return FruitEdge(cursor=strawberry.ID('1'), node_field=Fruit(name='Banana'))
    schema = strawberry.Schema(query=Query)
    query = '{\n        example {\n            __typename\n            cursor\n            nodeField {\n                name\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'example': {'__typename': 'FruitEdge', 'cursor': '1', 'nodeField': {'name': 'Banana'}}}

def test_supports_generic_specialized_with_list_type():
    if False:
        while True:
            i = 10
    T = TypeVar('T')

    @strawberry.type
    class Fruit:
        name: str

    @strawberry.type
    class Edge(Generic[T]):
        cursor: strawberry.ID
        nodes: List[T]

    @strawberry.type
    class FruitEdge(Edge[Fruit]):
        ...

    @strawberry.type
    class Query:

        @strawberry.field
        def example(self) -> FruitEdge:
            if False:
                return 10
            return FruitEdge(cursor=strawberry.ID('1'), nodes=[Fruit(name='Banana'), Fruit(name='Apple')])
    schema = strawberry.Schema(query=Query)
    query = '{\n        example {\n            __typename\n            cursor\n            nodes {\n                name\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'example': {'__typename': 'FruitEdge', 'cursor': '1', 'nodes': [{'name': 'Banana'}, {'name': 'Apple'}]}}

def test_supports_generic():
    if False:
        i = 10
        return i + 15
    T = TypeVar('T')

    @strawberry.type
    class Edge(Generic[T]):
        cursor: strawberry.ID
        node: T

    @strawberry.type
    class Person:
        name: str

    @strawberry.type
    class Query:

        @strawberry.field
        def example(self) -> Edge[Person]:
            if False:
                return 10
            return Edge(cursor=strawberry.ID('1'), node=Person(name='Example'))
    schema = strawberry.Schema(query=Query)
    query = '{\n        example {\n            __typename\n            cursor\n            node {\n                name\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'example': {'__typename': 'PersonEdge', 'cursor': '1', 'node': {'name': 'Example'}}}

def test_supports_multiple_generic():
    if False:
        print('Hello World!')
    A = TypeVar('A')
    B = TypeVar('B')

    @strawberry.type
    class Multiple(Generic[A, B]):
        a: A
        b: B

    @strawberry.type
    class Query:

        @strawberry.field
        def multiple(self) -> Multiple[int, str]:
            if False:
                i = 10
                return i + 15
            return Multiple(a=123, b='123')
    schema = strawberry.Schema(query=Query)
    query = '{\n        multiple {\n            __typename\n            a\n            b\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'multiple': {'__typename': 'IntStrMultiple', 'a': 123, 'b': '123'}}

def test_support_nested_generics():
    if False:
        return 10
    T = TypeVar('T')

    @strawberry.type
    class User:
        name: str

    @strawberry.type
    class Edge(Generic[T]):
        node: T

    @strawberry.type
    class Connection(Generic[T]):
        edge: Edge[T]

    @strawberry.type
    class Query:

        @strawberry.field
        def users(self) -> Connection[User]:
            if False:
                for i in range(10):
                    print('nop')
            return Connection(edge=Edge(node=User(name='Patrick')))
    schema = strawberry.Schema(query=Query)
    query = '{\n        users {\n            __typename\n            edge {\n                __typename\n                node {\n                    name\n                }\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'users': {'__typename': 'UserConnection', 'edge': {'__typename': 'UserEdge', 'node': {'name': 'Patrick'}}}}

def test_supports_optional():
    if False:
        for i in range(10):
            print('nop')
    T = TypeVar('T')

    @strawberry.type
    class User:
        name: str

    @strawberry.type
    class Edge(Generic[T]):
        node: Optional[T] = None

    @strawberry.type
    class Query:

        @strawberry.field
        def user(self) -> Edge[User]:
            if False:
                return 10
            return Edge()
    schema = strawberry.Schema(query=Query)
    query = '{\n        user {\n            __typename\n            node {\n                name\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'user': {'__typename': 'UserEdge', 'node': None}}

def test_supports_lists():
    if False:
        print('Hello World!')
    T = TypeVar('T')

    @strawberry.type
    class User:
        name: str

    @strawberry.type
    class Edge(Generic[T]):
        nodes: List[T]

    @strawberry.type
    class Query:

        @strawberry.field
        def user(self) -> Edge[User]:
            if False:
                return 10
            return Edge(nodes=[])
    schema = strawberry.Schema(query=Query)
    query = '{\n        user {\n            __typename\n            nodes {\n                name\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'user': {'__typename': 'UserEdge', 'nodes': []}}

def test_supports_lists_of_optionals():
    if False:
        print('Hello World!')
    T = TypeVar('T')

    @strawberry.type
    class User:
        name: str

    @strawberry.type
    class Edge(Generic[T]):
        nodes: List[Optional[T]]

    @strawberry.type
    class Query:

        @strawberry.field
        def user(self) -> Edge[User]:
            if False:
                i = 10
                return i + 15
            return Edge(nodes=[None])
    schema = strawberry.Schema(query=Query)
    query = '{\n        user {\n            __typename\n            nodes {\n                name\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'user': {'__typename': 'UserEdge', 'nodes': [None]}}

def test_can_extend_generics():
    if False:
        i = 10
        return i + 15
    T = TypeVar('T')

    @strawberry.type
    class User:
        name: str

    @strawberry.type
    class Edge(Generic[T]):
        node: T

    @strawberry.type
    class Connection(Generic[T]):
        edges: List[Edge[T]]

    @strawberry.type
    class ConnectionWithMeta(Connection[T]):
        meta: str

    @strawberry.type
    class Query:

        @strawberry.field
        def users(self) -> ConnectionWithMeta[User]:
            if False:
                return 10
            return ConnectionWithMeta(meta='123', edges=[Edge(node=User(name='Patrick'))])
    schema = strawberry.Schema(query=Query)
    query = '{\n        users {\n            __typename\n            meta\n            edges {\n                __typename\n                node {\n                    name\n                }\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'users': {'__typename': 'UserConnectionWithMeta', 'meta': '123', 'edges': [{'__typename': 'UserEdge', 'node': {'name': 'Patrick'}}]}}

def test_supports_generic_in_unions():
    if False:
        i = 10
        return i + 15
    T = TypeVar('T')

    @strawberry.type
    class Edge(Generic[T]):
        cursor: strawberry.ID
        node: T

    @strawberry.type
    class Fallback:
        node: str

    @strawberry.type
    class Query:

        @strawberry.field
        def example(self) -> Union[Fallback, Edge[int]]:
            if False:
                while True:
                    i = 10
            return Edge(cursor=strawberry.ID('1'), node=1)
    schema = strawberry.Schema(query=Query)
    query = '{\n        example {\n            __typename\n\n            ... on IntEdge {\n                cursor\n                node\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'example': {'__typename': 'IntEdge', 'cursor': '1', 'node': 1}}

def test_generic_with_enum_as_param_of_type_inside_unions():
    if False:
        while True:
            i = 10
    T = TypeVar('T')

    @strawberry.type
    class Pet:
        name: str

    @strawberry.type
    class ErrorNode(Generic[T]):
        code: T

    @strawberry.enum
    class Codes(Enum):
        a = 'a'
        b = 'b'

    @strawberry.type
    class Query:

        @strawberry.field
        def result(self) -> Union[Pet, ErrorNode[Codes]]:
            if False:
                i = 10
                return i + 15
            return ErrorNode(code=Codes.a)
    schema = strawberry.Schema(query=Query)
    query = '{\n        result {\n            __typename\n            ... on CodesErrorNode {\n                code\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'result': {'__typename': 'CodesErrorNode', 'code': 'a'}}

def test_generic_with_enum():
    if False:
        return 10
    T = TypeVar('T')

    @strawberry.enum
    class EstimatedValueEnum(Enum):
        test = 'test'
        testtest = 'testtest'

    @strawberry.type
    class EstimatedValue(Generic[T]):
        value: T
        type: EstimatedValueEnum

    @strawberry.type
    class Query:

        @strawberry.field
        def estimated_value(self) -> Optional[EstimatedValue[int]]:
            if False:
                return 10
            return EstimatedValue(value=1, type=EstimatedValueEnum.test)
    schema = strawberry.Schema(query=Query)
    query = '{\n        estimatedValue {\n            __typename\n            value\n            type\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'estimatedValue': {'__typename': 'IntEstimatedValue', 'value': 1, 'type': 'test'}}

def test_supports_generic_in_unions_multiple_vars():
    if False:
        return 10
    A = TypeVar('A')
    B = TypeVar('B')

    @strawberry.type
    class Edge(Generic[A, B]):
        info: A
        node: B

    @strawberry.type
    class Fallback:
        node: str

    @strawberry.type
    class Query:

        @strawberry.field
        def example(self) -> Union[Fallback, Edge[int, str]]:
            if False:
                print('Hello World!')
            return Edge(node='string', info=1)
    schema = strawberry.Schema(query=Query)
    query = '{\n        example {\n            __typename\n\n            ... on IntStrEdge {\n                node\n                info\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'example': {'__typename': 'IntStrEdge', 'node': 'string', 'info': 1}}

def test_supports_generic_in_unions_with_nesting():
    if False:
        for i in range(10):
            print('nop')
    T = TypeVar('T')

    @strawberry.type
    class User:
        name: str

    @strawberry.type
    class Edge(Generic[T]):
        node: T

    @strawberry.type
    class Connection(Generic[T]):
        edge: Edge[T]

    @strawberry.type
    class Fallback:
        node: str

    @strawberry.type
    class Query:

        @strawberry.field
        def users(self) -> Union[Connection[User], Fallback]:
            if False:
                for i in range(10):
                    print('nop')
            return Connection(edge=Edge(node=User(name='Patrick')))
    schema = strawberry.Schema(query=Query)
    query = '{\n        users {\n            __typename\n            ... on UserConnection {\n                edge {\n                    __typename\n                    node {\n                        name\n                    }\n                }\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'users': {'__typename': 'UserConnection', 'edge': {'__typename': 'UserEdge', 'node': {'name': 'Patrick'}}}}

def test_supports_multiple_generics_in_union():
    if False:
        print('Hello World!')
    T = TypeVar('T')

    @strawberry.type
    class Edge(Generic[T]):
        cursor: strawberry.ID
        node: T

    @strawberry.type
    class Query:

        @strawberry.field
        def example(self) -> List[Union[Edge[int], Edge[str]]]:
            if False:
                while True:
                    i = 10
            return [Edge(cursor=strawberry.ID('1'), node=1), Edge(cursor=strawberry.ID('2'), node='string')]
    schema = strawberry.Schema(query=Query)
    expected_schema = '\n      type IntEdge {\n        cursor: ID!\n        node: Int!\n      }\n\n      union IntEdgeStrEdge = IntEdge | StrEdge\n\n      type Query {\n        example: [IntEdgeStrEdge!]!\n      }\n\n      type StrEdge {\n        cursor: ID!\n        node: String!\n      }\n    '
    assert str(schema) == textwrap.dedent(expected_schema).strip()
    query = '{\n        example {\n            __typename\n\n            ... on IntEdge {\n                cursor\n                intNode: node\n            }\n\n            ... on StrEdge {\n                cursor\n                strNode: node\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'example': [{'__typename': 'IntEdge', 'cursor': '1', 'intNode': 1}, {'__typename': 'StrEdge', 'cursor': '2', 'strNode': 'string'}]}

def test_generics_via_anonymous_union():
    if False:
        i = 10
        return i + 15
    T = TypeVar('T')

    @strawberry.type
    class Edge(Generic[T]):
        cursor: str
        node: T

    @strawberry.type
    class Connection(Generic[T]):
        edges: List[Edge[T]]

    @strawberry.type
    class Entity1:
        id: int

    @strawberry.type
    class Entity2:
        id: int

    @strawberry.type
    class Query:
        entities: Connection[Union[Entity1, Entity2]]
    schema = strawberry.Schema(query=Query)
    expected_schema = textwrap.dedent('\n        type Entity1 {\n          id: Int!\n        }\n\n        union Entity1Entity2 = Entity1 | Entity2\n\n        type Entity1Entity2Connection {\n          edges: [Entity1Entity2Edge!]!\n        }\n\n        type Entity1Entity2Edge {\n          cursor: String!\n          node: Entity1Entity2!\n        }\n\n        type Entity2 {\n          id: Int!\n        }\n\n        type Query {\n          entities: Entity1Entity2Connection!\n        }\n        ').strip()
    assert str(schema) == expected_schema

def test_generated_names():
    if False:
        for i in range(10):
            print('nop')
    T = TypeVar('T')

    @strawberry.type
    class EdgeWithCursor(Generic[T]):
        cursor: strawberry.ID
        node: T

    @strawberry.type
    class SpecialPerson:
        name: str

    @strawberry.type
    class Query:

        @strawberry.field
        def person_edge(self) -> EdgeWithCursor[SpecialPerson]:
            if False:
                for i in range(10):
                    print('nop')
            return EdgeWithCursor(cursor=strawberry.ID('1'), node=SpecialPerson(name='Example'))
    schema = strawberry.Schema(query=Query)
    query = '{\n        personEdge {\n            __typename\n            cursor\n            node {\n                name\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'personEdge': {'__typename': 'SpecialPersonEdgeWithCursor', 'cursor': '1', 'node': {'name': 'Example'}}}

def test_supports_lists_within_unions():
    if False:
        print('Hello World!')
    T = TypeVar('T')

    @strawberry.type
    class User:
        name: str

    @strawberry.type
    class Edge(Generic[T]):
        nodes: List[T]

    @strawberry.type
    class Query:

        @strawberry.field
        def user(self) -> Union[User, Edge[User]]:
            if False:
                for i in range(10):
                    print('nop')
            return Edge(nodes=[User(name='P')])
    schema = strawberry.Schema(query=Query)
    query = '{\n        user {\n            __typename\n\n            ... on UserEdge {\n                nodes {\n                    name\n                }\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'user': {'__typename': 'UserEdge', 'nodes': [{'name': 'P'}]}}

def test_supports_lists_within_unions_empty_list():
    if False:
        i = 10
        return i + 15
    T = TypeVar('T')

    @strawberry.type
    class User:
        name: str

    @strawberry.type
    class Edge(Generic[T]):
        nodes: List[T]

    @strawberry.type
    class Query:

        @strawberry.field
        def user(self) -> Union[User, Edge[User]]:
            if False:
                print('Hello World!')
            return Edge(nodes=[])
    schema = strawberry.Schema(query=Query)
    query = '{\n        user {\n            __typename\n\n            ... on UserEdge {\n                nodes {\n                    name\n                }\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'user': {'__typename': 'UserEdge', 'nodes': []}}

@pytest.mark.xfail()
def test_raises_error_when_unable_to_find_type():
    if False:
        i = 10
        return i + 15
    T = TypeVar('T')

    @strawberry.type
    class User:
        name: str

    @strawberry.type
    class Edge(Generic[T]):
        nodes: List[T]

    @strawberry.type
    class Query:

        @strawberry.field
        def user(self) -> Union[User, Edge[User]]:
            if False:
                while True:
                    i = 10
            return Edge(nodes=['bad example'])
    schema = strawberry.Schema(query=Query)
    query = '{\n        user {\n            __typename\n\n            ... on UserEdge {\n                nodes {\n                    name\n                }\n            }\n        }\n    }'
    result = schema.execute_sync(query)
    assert result.errors[0].message == "Unable to find type for <class 'tests.schema.test_generics.test_raises_error_when_unable_to_find_type.<locals>.Edge'> and (<class 'str'>,)"

def test_generic_with_arguments():
    if False:
        while True:
            i = 10
    T = TypeVar('T')

    @strawberry.type
    class Collection(Generic[T]):

        @strawberry.field
        def by_id(self, ids: List[int]) -> List[T]:
            if False:
                while True:
                    i = 10
            return []

    @strawberry.type
    class Post:
        name: str

    @strawberry.type
    class Query:
        user: Collection[Post]
    schema = strawberry.Schema(Query)
    expected_schema = '\n    type Post {\n      name: String!\n    }\n\n    type PostCollection {\n      byId(ids: [Int!]!): [Post!]!\n    }\n\n    type Query {\n      user: PostCollection!\n    }\n    '
    assert str(schema) == textwrap.dedent(expected_schema).strip()

def test_generic_argument():
    if False:
        for i in range(10):
            print('nop')
    T = TypeVar('T')

    @strawberry.type
    class Node(Generic[T]):

        @strawberry.field
        def edge(self, arg: T) -> bool:
            if False:
                print('Hello World!')
            return bool(arg)

        @strawberry.field
        def edges(self, args: List[T]) -> int:
            if False:
                return 10
            return len(args)

    @strawberry.type
    class Query:
        i_node: Node[int]
        b_node: Node[bool]
    schema = strawberry.Schema(Query)
    expected_schema = '\n    type BoolNode {\n      edge(arg: Boolean!): Boolean!\n      edges(args: [Boolean!]!): Int!\n    }\n\n    type IntNode {\n      edge(arg: Int!): Boolean!\n      edges(args: [Int!]!): Int!\n    }\n\n    type Query {\n      iNode: IntNode!\n      bNode: BoolNode!\n    }\n    '
    assert str(schema) == textwrap.dedent(expected_schema).strip()

def test_generic_extra_type():
    if False:
        print('Hello World!')
    T = TypeVar('T')

    @strawberry.type
    class Node(Generic[T]):
        field: T

    @strawberry.type
    class Query:
        name: str
    schema = strawberry.Schema(Query, types=[Node[int]])
    expected_schema = '\n    type IntNode {\n      field: Int!\n    }\n\n    type Query {\n      name: String!\n    }\n    '
    assert str(schema) == textwrap.dedent(expected_schema).strip()

def test_generic_extending_with_type_var():
    if False:
        print('Hello World!')
    T = TypeVar('T')

    @strawberry.interface
    class Node(Generic[T]):
        id: strawberry.ID

        def _resolve(self) -> Optional[T]:
            if False:
                for i in range(10):
                    print('nop')
            return None

    @strawberry.type
    class Book(Node[str]):
        name: str

    @strawberry.type
    class Query:

        @strawberry.field
        def books(self) -> List[Book]:
            if False:
                print('Hello World!')
            return list()
    schema = strawberry.Schema(query=Query)
    expected_schema = '\n    type Book implements Node {\n      id: ID!\n      name: String!\n    }\n\n    interface Node {\n      id: ID!\n    }\n\n    type Query {\n      books: [Book!]!\n    }\n    '
    assert str(schema) == textwrap.dedent(expected_schema).strip()

def test_self():
    if False:
        while True:
            i = 10

    @strawberry.interface
    class INode:
        field: Optional[Self]
        fields: List[Self]

    @strawberry.type
    class Node(INode):
        ...
    schema = strawberry.Schema(query=Node)
    expected_schema = '\n    schema {\n      query: Node\n    }\n\n    interface INode {\n      field: INode\n      fields: [INode!]!\n    }\n\n    type Node implements INode {\n      field: Node\n      fields: [Node!]!\n    }\n    '
    assert str(schema) == textwrap.dedent(expected_schema).strip()
    query = '{\n        field {\n            __typename\n        }\n        fields {\n            __typename\n        }\n    }'
    result = schema.execute_sync(query, root_value=Node(field=None, fields=[]))
    assert result.data == {'field': None, 'fields': []}

def test_supports_generic_input_type():
    if False:
        while True:
            i = 10
    T = TypeVar('T')

    @strawberry.input
    class Input(Generic[T]):
        field: T

    @strawberry.type
    class Query:

        @strawberry.field
        def field(self, input: Input[str]) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return input.field
    schema = strawberry.Schema(query=Query)
    query = '{\n        field(input: { field: "data" })\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'field': 'data'}

def test_generic_interface():
    if False:
        return 10

    @strawberry.interface
    class ObjectType:
        obj: strawberry.Private[Any]

        @strawberry.field
        def repr(self) -> str:
            if False:
                i = 10
                return i + 15
            return str(self.obj)
    T = TypeVar('T')

    @strawberry.type
    class GenericObject(ObjectType, Generic[T]):

        @strawberry.field
        def value(self) -> T:
            if False:
                for i in range(10):
                    print('nop')
            return self.obj

    @strawberry.type
    class Query:

        @strawberry.field
        def foo(self) -> GenericObject[str]:
            if False:
                print('Hello World!')
            return GenericObject(obj='foo')
    schema = strawberry.Schema(query=Query)
    query_result = schema.execute_sync('\n            query {\n                foo {\n                    __typename\n                    value\n                    repr\n                }\n            }\n        ')
    assert not query_result.errors
    assert query_result.data == {'foo': {'__typename': 'StrGenericObject', 'value': 'foo', 'repr': 'foo'}}

def test_generic_interface_extra_types():
    if False:
        print('Hello World!')
    T = TypeVar('T')

    @strawberry.interface
    class Abstract:
        x: str = ''

    @strawberry.type
    class Real(Generic[T], Abstract):
        y: T

    @strawberry.type
    class Query:

        @strawberry.field
        def real(self) -> Abstract:
            if False:
                i = 10
                return i + 15
            return Real[int](y=0)
    schema = strawberry.Schema(Query, types=[Real[int]])
    assert str(schema) == textwrap.dedent('\n            interface Abstract {\n              x: String!\n            }\n\n            type IntReal implements Abstract {\n              x: String!\n              y: Int!\n            }\n\n            type Query {\n              real: Abstract!\n            }\n            ').strip()
    query_result = schema.execute_sync('{ real { __typename x } }')
    assert not query_result.errors
    assert query_result.data == {'real': {'__typename': 'IntReal', 'x': ''}}