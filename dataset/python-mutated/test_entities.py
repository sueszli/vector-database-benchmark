import textwrap
from typing import List
import strawberry

def test_entities_type_when_no_type_has_keys():
    if False:
        return 10
    global Review

    @strawberry.federation.type
    class User:
        username: str

    @strawberry.federation.type(extend=True)
    class Product:
        upc: str = strawberry.federation.field(external=True)
        reviews: List['Review']

    @strawberry.federation.type
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
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@external"]) {\n          query: Query\n        }\n\n        extend type Product {\n          upc: String! @external\n          reviews: [Review!]!\n        }\n\n        type Query {\n          _service: _Service!\n          topProducts(first: Int!): [Product!]!\n        }\n\n        type Review {\n          body: String!\n          author: User!\n          product: Product!\n        }\n\n        type User {\n          username: String!\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()
    del Review

def test_entities_type_when_one_type_has_keys():
    if False:
        for i in range(10):
            print('nop')
    global Review

    @strawberry.federation.type
    class User:
        username: str

    @strawberry.federation.type(keys=['upc'], extend=True)
    class Product:
        upc: str = strawberry.federation.field(external=True)
        reviews: List['Review']

    @strawberry.federation.type
    class Review:
        body: str
        author: User
        product: Product

    @strawberry.federation.type
    class Query:

        @strawberry.field
        def top_products(self, first: int) -> List[Product]:
            if False:
                for i in range(10):
                    print('nop')
            return []
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@external", "@key"]) {\n          query: Query\n        }\n\n        extend type Product @key(fields: "upc") {\n          upc: String! @external\n          reviews: [Review!]!\n        }\n\n        type Query {\n          _entities(representations: [_Any!]!): [_Entity]!\n          _service: _Service!\n          topProducts(first: Int!): [Product!]!\n        }\n\n        type Review {\n          body: String!\n          author: User!\n          product: Product!\n        }\n\n        type User {\n          username: String!\n        }\n\n        scalar _Any\n\n        union _Entity = Product\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()
    del Review