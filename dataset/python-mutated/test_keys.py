import textwrap
from typing import List
import strawberry
from strawberry.federation.schema_directives import Key

def test_keys_federation_1():
    if False:
        while True:
            i = 10
    global Review

    @strawberry.federation.type
    class User:
        username: str

    @strawberry.federation.type(keys=[Key(fields='upc', resolvable=True)], extend=True)
    class Product:
        upc: str = strawberry.federation.field(external=True)
        reviews: List['Review']

    @strawberry.federation.type(keys=['body'])
    class Review:
        body: str
        author: User
        product: Product

    @strawberry.federation.type
    class Query:

        @strawberry.field
        def top_products(self, first: int) -> List[Product]:
            if False:
                while True:
                    i = 10
            return []
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=False)
    expected = '\n        extend type Product @key(fields: "upc") {\n          upc: String! @external\n          reviews: [Review!]!\n        }\n\n        type Query {\n          _entities(representations: [_Any!]!): [_Entity]!\n          _service: _Service!\n          topProducts(first: Int!): [Product!]!\n        }\n\n        type Review @key(fields: "body") {\n          body: String!\n          author: User!\n          product: Product!\n        }\n\n        type User {\n          username: String!\n        }\n\n        scalar _Any\n\n        union _Entity = Product | Review\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()
    del Review

def test_keys_federation_2():
    if False:
        return 10
    global Review

    @strawberry.federation.type
    class User:
        username: str

    @strawberry.federation.type(keys=[Key(fields='upc', resolvable=True)], extend=True)
    class Product:
        upc: str = strawberry.federation.field(external=True)
        reviews: List['Review']

    @strawberry.federation.type(keys=['body'])
    class Review:
        body: str
        author: User
        product: Product

    @strawberry.federation.type
    class Query:

        @strawberry.field
        def top_products(self, first: int) -> List[Product]:
            if False:
                print('Hello World!')
            return []
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@external", "@key"]) {\n          query: Query\n        }\n\n        extend type Product @key(fields: "upc", resolvable: true) {\n          upc: String! @external\n          reviews: [Review!]!\n        }\n\n        type Query {\n          _entities(representations: [_Any!]!): [_Entity]!\n          _service: _Service!\n          topProducts(first: Int!): [Product!]!\n        }\n\n        type Review @key(fields: "body") {\n          body: String!\n          author: User!\n          product: Product!\n        }\n\n        type User {\n          username: String!\n        }\n\n        scalar _Any\n\n        union _Entity = Product | Review\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()
    del Review