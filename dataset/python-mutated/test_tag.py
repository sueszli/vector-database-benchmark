import textwrap
from enum import Enum
from typing import List
from typing_extensions import Annotated
import strawberry

def test_field_tag_printed_correctly():
    if False:
        print('Hello World!')

    @strawberry.federation.interface(tags=['myTag', 'anotherTag'])
    class SomeInterface:
        id: strawberry.ID

    @strawberry.federation.type(tags=['myTag', 'anotherTag'])
    class Product(SomeInterface):
        upc: str = strawberry.federation.field(external=True, tags=['myTag', 'anotherTag'])

    @strawberry.federation.type
    class Query:

        @strawberry.field
        def top_products(self, first: Annotated[int, strawberry.federation.argument(tags=['myTag'])]) -> List[Product]:
            if False:
                i = 10
                return i + 15
            return []
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@external", "@tag"]) {\n          query: Query\n        }\n\n        type Product implements SomeInterface @tag(name: "myTag") @tag(name: "anotherTag") {\n          id: ID!\n          upc: String! @external @tag(name: "myTag") @tag(name: "anotherTag")\n        }\n\n        type Query {\n          _service: _Service!\n          topProducts(first: Int! @tag(name: "myTag")): [Product!]!\n        }\n\n        interface SomeInterface @tag(name: "myTag") @tag(name: "anotherTag") {\n          id: ID!\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_field_tag_printed_correctly_on_scalar():
    if False:
        return 10

    @strawberry.federation.scalar(tags=['myTag', 'anotherTag'])
    class SomeScalar(str):
        __slots__ = ()

    @strawberry.federation.type
    class Query:
        hello: SomeScalar
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@tag"]) {\n          query: Query\n        }\n\n        type Query {\n          _service: _Service!\n          hello: SomeScalar!\n        }\n\n        scalar SomeScalar @tag(name: "myTag") @tag(name: "anotherTag")\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_field_tag_printed_correctly_on_enum():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.federation.enum(tags=['myTag', 'anotherTag'])
    class SomeEnum(Enum):
        A = 'A'

    @strawberry.federation.type
    class Query:
        hello: SomeEnum
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@tag"]) {\n          query: Query\n        }\n\n        type Query {\n          _service: _Service!\n          hello: SomeEnum!\n        }\n\n        enum SomeEnum @tag(name: "myTag") @tag(name: "anotherTag") {\n          A\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_field_tag_printed_correctly_on_enum_value():
    if False:
        while True:
            i = 10

    @strawberry.enum
    class SomeEnum(Enum):
        A = strawberry.federation.enum_value('A', tags=['myTag', 'anotherTag'])

    @strawberry.federation.type
    class Query:
        hello: SomeEnum
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@tag"]) {\n          query: Query\n        }\n\n        type Query {\n          _service: _Service!\n          hello: SomeEnum!\n        }\n\n        enum SomeEnum {\n          A @tag(name: "myTag") @tag(name: "anotherTag")\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_field_tag_printed_correctly_on_union():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class A:
        a: str

    @strawberry.type
    class B:
        b: str
    Union = strawberry.federation.union('Union', (A, B), tags=['myTag', 'anotherTag'])

    @strawberry.federation.type
    class Query:
        hello: Union
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@tag"]) {\n          query: Query\n        }\n\n        type A {\n          a: String!\n        }\n\n        type B {\n          b: String!\n        }\n\n        type Query {\n          _service: _Service!\n          hello: Union!\n        }\n\n        union Union @tag(name: "myTag") @tag(name: "anotherTag") = A | B\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_tag_printed_correctly_on_inputs():
    if False:
        i = 10
        return i + 15

    @strawberry.federation.input(tags=['myTag', 'anotherTag'])
    class Input:
        a: str = strawberry.federation.field(tags=['myTag', 'anotherTag'])

    @strawberry.federation.type
    class Query:
        hello: str
    schema = strawberry.federation.Schema(query=Query, types=[Input], enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@tag"]) {\n          query: Query\n        }\n\n        input Input @tag(name: "myTag") @tag(name: "anotherTag") {\n          a: String! @tag(name: "myTag") @tag(name: "anotherTag")\n        }\n\n        type Query {\n          _service: _Service!\n          hello: String!\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()