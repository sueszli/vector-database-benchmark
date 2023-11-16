import textwrap
import strawberry
from strawberry.schema_directive import Location

def test_schema_directives_and_compose_schema():
    if False:
        while True:
            i = 10

    @strawberry.federation.schema_directive(locations=[Location.OBJECT], name='cacheControl', compose=True)
    class CacheControl:
        max_age: int

    @strawberry.federation.schema_directive(locations=[Location.OBJECT], name='sensitive')
    class Sensitive:
        reason: str

    @strawberry.federation.type(keys=['id'], shareable=True, extend=True, directives=[CacheControl(max_age=42), Sensitive(reason='example')])
    class FederatedType:
        id: strawberry.ID

    @strawberry.type
    class Query:
        federatedType: FederatedType
    expected_type = '\n    directive @cacheControl(maxAge: Int!) on OBJECT\n\n    directive @sensitive(reason: String!) on OBJECT\n\n    schema @composeDirective(name: "@cacheControl") @link(url: "https://directives.strawberry.rocks/cacheControl/v0.1", import: ["@cacheControl"]) @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@composeDirective", "@key", "@shareable"]) {\n      query: Query\n    }\n\n    extend type FederatedType @cacheControl(maxAge: 42) @sensitive(reason: "example") @key(fields: "id") @shareable {\n      id: ID!\n    }\n\n    type Query {\n      _entities(representations: [_Any!]!): [_Entity]!\n      _service: _Service!\n      federatedType: FederatedType!\n    }\n\n    scalar _Any\n\n    union _Entity = FederatedType\n\n    type _Service {\n      sdl: String!\n    }\n    '
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    assert schema.as_str() == textwrap.dedent(expected_type).strip()

def test_schema_directives_and_compose_schema_custom_import_url():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.federation.schema_directive(locations=[Location.OBJECT], name='cacheControl', compose=True, import_url='https://f.strawberry.rocks/cacheControl/v1.0')
    class CacheControl:
        max_age: int

    @strawberry.federation.schema_directive(locations=[Location.OBJECT], name='sensitive')
    class Sensitive:
        reason: str

    @strawberry.federation.type(keys=['id'], shareable=True, extend=True, directives=[CacheControl(max_age=42), Sensitive(reason='example')])
    class FederatedType:
        id: strawberry.ID

    @strawberry.type
    class Query:
        federatedType: FederatedType
    expected_type = '\n    directive @cacheControl(maxAge: Int!) on OBJECT\n\n    directive @sensitive(reason: String!) on OBJECT\n\n    schema @composeDirective(name: "@cacheControl") @link(url: "https://f.strawberry.rocks/cacheControl/v1.0", import: ["@cacheControl"]) @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@composeDirective", "@key", "@shareable"]) {\n      query: Query\n    }\n\n    extend type FederatedType @cacheControl(maxAge: 42) @sensitive(reason: "example") @key(fields: "id") @shareable {\n      id: ID!\n    }\n\n    type Query {\n      _entities(representations: [_Any!]!): [_Entity]!\n      _service: _Service!\n      federatedType: FederatedType!\n    }\n\n    scalar _Any\n\n    union _Entity = FederatedType\n\n    type _Service {\n      sdl: String!\n    }\n    '
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    assert schema.as_str() == textwrap.dedent(expected_type).strip()