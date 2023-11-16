import textwrap
from typing import List
import strawberry

def test_fields_requires_are_printed_correctly():
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
        field1: str = strawberry.federation.field(external=True)
        field2: str = strawberry.federation.field(external=True)
        field3: str = strawberry.federation.field(external=True)

        @strawberry.federation.field(requires=['field1', 'field2', 'field3'])
        def reviews(self) -> List['Review']:
            if False:
                while True:
                    i = 10
            return []

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
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@external", "@key", "@requires"]) {\n          query: Query\n        }\n\n        extend type Product @key(fields: "upc") {\n          upc: String! @external\n          field1: String! @external\n          field2: String! @external\n          field3: String! @external\n          reviews: [Review!]! @requires(fields: "field1 field2 field3")\n        }\n\n        type Query {\n          _entities(representations: [_Any!]!): [_Entity]!\n          _service: _Service!\n          topProducts(first: Int!): [Product!]!\n        }\n\n        type Review {\n          body: String!\n          author: User!\n          product: Product!\n        }\n\n        type User {\n          username: String!\n        }\n\n        scalar _Any\n\n        union _Entity = Product\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()
    del Review