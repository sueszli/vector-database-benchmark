import textwrap
import strawberry

def test_interface_object():
    if False:
        while True:
            i = 10

    @strawberry.federation.interface_object(keys=['id'])
    class SomeInterface:
        id: strawberry.ID
    schema = strawberry.federation.Schema(types=[SomeInterface], enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@interfaceObject", "@key"]) {\n          query: Query\n        }\n\n        type Query {\n          _entities(representations: [_Any!]!): [_Entity]!\n          _service: _Service!\n        }\n\n        type SomeInterface @key(fields: "id") @interfaceObject {\n          id: ID!\n        }\n\n        scalar _Any\n\n        union _Entity = SomeInterface\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()