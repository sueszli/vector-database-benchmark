import textwrap
from typing import List
import strawberry
from strawberry.schema.config import StrawberryConfig

def test_field_provides_are_printed_correctly_camel_case_on():
    if False:
        i = 10
        return i + 15
    global Review

    @strawberry.federation.type
    class User:
        username: str

    @strawberry.federation.type(keys=['upc'], extend=True)
    class Product:
        upc: str = strawberry.federation.field(external=True)
        the_name: str = strawberry.federation.field(external=True)
        reviews: List['Review']

    @strawberry.federation.type
    class Review:
        body: str
        author: User
        product: Product = strawberry.federation.field(provides=['name'])

    @strawberry.federation.type
    class Query:

        @strawberry.field
        def top_products(self, first: int) -> List[Product]:
            if False:
                for i in range(10):
                    print('nop')
            return []
    schema = strawberry.federation.Schema(query=Query, config=StrawberryConfig(auto_camel_case=True), enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@external", "@key", "@provides"]) {\n          query: Query\n        }\n\n        extend type Product @key(fields: "upc") {\n          upc: String! @external\n          theName: String! @external\n          reviews: [Review!]!\n        }\n\n        type Query {\n          _entities(representations: [_Any!]!): [_Entity]!\n          _service: _Service!\n          topProducts(first: Int!): [Product!]!\n        }\n\n        type Review {\n          body: String!\n          author: User!\n          product: Product! @provides(fields: "name")\n        }\n\n        type User {\n          username: String!\n        }\n\n        scalar _Any\n\n        union _Entity = Product\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()
    del Review

def test_field_provides_are_printed_correctly_camel_case_off():
    if False:
        i = 10
        return i + 15
    global Review

    @strawberry.federation.type
    class User:
        username: str

    @strawberry.federation.type(keys=['upc'], extend=True)
    class Product:
        upc: str = strawberry.federation.field(external=True)
        the_name: str = strawberry.federation.field(external=True)
        reviews: List['Review']

    @strawberry.federation.type
    class Review:
        body: str
        author: User
        product: Product = strawberry.federation.field(provides=['name'])

    @strawberry.federation.type
    class Query:

        @strawberry.field
        def top_products(self, first: int) -> List[Product]:
            if False:
                for i in range(10):
                    print('nop')
            return []
    schema = strawberry.federation.Schema(query=Query, config=StrawberryConfig(auto_camel_case=False), enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@external", "@key", "@provides"]) {\n          query: Query\n        }\n\n        extend type Product @key(fields: "upc") {\n          upc: String! @external\n          the_name: String! @external\n          reviews: [Review!]!\n        }\n\n        type Query {\n          _entities(representations: [_Any!]!): [_Entity]!\n          _service: _Service!\n          top_products(first: Int!): [Product!]!\n        }\n\n        type Review {\n          body: String!\n          author: User!\n          product: Product! @provides(fields: "name")\n        }\n\n        type User {\n          username: String!\n        }\n\n        scalar _Any\n\n        union _Entity = Product\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()
    del Review