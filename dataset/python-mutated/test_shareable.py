import textwrap
from typing import List
import strawberry

def test_field_shareable_printed_correctly():
    if False:
        i = 10
        return i + 15

    @strawberry.interface
    class SomeInterface:
        id: strawberry.ID

    @strawberry.federation.type(keys=['upc'], extend=True, shareable=True)
    class Product(SomeInterface):
        upc: str = strawberry.federation.field(external=True, shareable=True)

    @strawberry.federation.type
    class Query:

        @strawberry.field
        def top_products(self, first: int) -> List[Product]:
            if False:
                while True:
                    i = 10
            return []
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@external", "@key", "@shareable"]) {\n          query: Query\n        }\n\n        extend type Product implements SomeInterface @key(fields: "upc") @shareable {\n          id: ID!\n          upc: String! @external @shareable\n        }\n\n        type Query {\n          _entities(representations: [_Any!]!): [_Entity]!\n          _service: _Service!\n          topProducts(first: Int!): [Product!]!\n        }\n\n        interface SomeInterface {\n          id: ID!\n        }\n\n        scalar _Any\n\n        union _Entity = Product\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()