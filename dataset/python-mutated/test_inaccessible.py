import textwrap
from enum import Enum
from typing import List
from typing_extensions import Annotated
import strawberry

def test_field_inaccessible_printed_correctly():
    if False:
        i = 10
        return i + 15

    @strawberry.federation.interface(inaccessible=True)
    class AnInterface:
        id: strawberry.ID

    @strawberry.interface
    class SomeInterface:
        id: strawberry.ID
        a_field: str = strawberry.federation.field(inaccessible=True)

    @strawberry.federation.type(keys=['upc'], extend=True)
    class Product(SomeInterface):
        upc: str = strawberry.federation.field(external=True, inaccessible=True)

    @strawberry.federation.input(inaccessible=True)
    class AnInput:
        id: strawberry.ID = strawberry.federation.field(inaccessible=True)

    @strawberry.federation.type(inaccessible=True)
    class AnInaccessibleType:
        id: strawberry.ID

    @strawberry.federation.type
    class Query:

        @strawberry.field
        def top_products(self, first: Annotated[int, strawberry.federation.argument(inaccessible=True)]) -> List[Product]:
            if False:
                while True:
                    i = 10
            return []
    schema = strawberry.federation.Schema(query=Query, types=[AnInterface, AnInput, AnInaccessibleType], enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@external", "@inaccessible", "@key"]) {\n          query: Query\n        }\n\n        type AnInaccessibleType @inaccessible {\n          id: ID!\n        }\n\n        input AnInput @inaccessible {\n          id: ID! @inaccessible\n        }\n\n        interface AnInterface @inaccessible {\n          id: ID!\n        }\n\n        extend type Product implements SomeInterface @key(fields: "upc") {\n          id: ID!\n          aField: String! @inaccessible\n          upc: String! @external @inaccessible\n        }\n\n        type Query {\n          _entities(representations: [_Any!]!): [_Entity]!\n          _service: _Service!\n          topProducts(first: Int! @inaccessible): [Product!]!\n        }\n\n        interface SomeInterface {\n          id: ID!\n          aField: String! @inaccessible\n        }\n\n        scalar _Any\n\n        union _Entity = Product\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_inaccessible_on_mutation():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class Query:
        hello: str

    @strawberry.type
    class Mutation:

        @strawberry.federation.mutation(inaccessible=True)
        def hello(self) -> str:
            if False:
                return 10
            return 'Hello'
    schema = strawberry.federation.Schema(query=Query, mutation=Mutation, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@inaccessible"]) {\n          query: Query\n          mutation: Mutation\n        }\n\n        type Mutation {\n          hello: String! @inaccessible\n        }\n\n        type Query {\n          _service: _Service!\n          hello: String!\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_inaccessible_on_scalar():
    if False:
        return 10
    SomeScalar = strawberry.federation.scalar(str, name='SomeScalar', inaccessible=True)

    @strawberry.type
    class Query:
        hello: SomeScalar
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@inaccessible"]) {\n          query: Query\n        }\n\n        type Query {\n          _service: _Service!\n          hello: SomeScalar!\n        }\n\n        scalar SomeScalar @inaccessible\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_inaccessible_on_enum():
    if False:
        return 10

    @strawberry.federation.enum(inaccessible=True)
    class SomeEnum(Enum):
        A = 'A'

    @strawberry.type
    class Query:
        hello: SomeEnum
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@inaccessible"]) {\n          query: Query\n        }\n\n        type Query {\n          _service: _Service!\n          hello: SomeEnum!\n        }\n\n        enum SomeEnum @inaccessible {\n          A\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_inaccessible_on_enum_value():
    if False:
        while True:
            i = 10

    @strawberry.enum
    class SomeEnum(Enum):
        A = strawberry.federation.enum_value('A', inaccessible=True)

    @strawberry.type
    class Query:
        hello: SomeEnum
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@inaccessible"]) {\n          query: Query\n        }\n\n        type Query {\n          _service: _Service!\n          hello: SomeEnum!\n        }\n\n        enum SomeEnum {\n          A @inaccessible\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_field_tag_printed_correctly_on_union():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class A:
        a: str

    @strawberry.type
    class B:
        b: str
    Union = strawberry.federation.union('Union', (A, B), inaccessible=True)

    @strawberry.federation.type
    class Query:
        hello: Union
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@inaccessible"]) {\n          query: Query\n        }\n\n        type A {\n          a: String!\n        }\n\n        type B {\n          b: String!\n        }\n\n        type Query {\n          _service: _Service!\n          hello: Union!\n        }\n\n        union Union @inaccessible = A | B\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()