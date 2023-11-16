import textwrap
import strawberry
from strawberry.federation.schema_directives import Link

def test_link_directive():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class Query:
        hello: str
    schema = strawberry.federation.Schema(query=Query, schema_directives=[Link(url='https://specs.apollo.dev/link/v1.0')])
    expected = '\n        schema @link(url: "https://specs.apollo.dev/link/v1.0") {\n          query: Query\n        }\n\n        type Query {\n          _service: _Service!\n          hello: String!\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_link_directive_imports():
    if False:
        return 10

    @strawberry.type
    class Query:
        hello: str
    schema = strawberry.federation.Schema(query=Query, schema_directives=[Link(url='https://specs.apollo.dev/federation/v2.3', import_=['@key', '@requires', '@provides', '@external', {'name': '@tag', 'as': '@mytag'}, '@extends', '@shareable', '@inaccessible', '@override'])])
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@key", "@requires", "@provides", "@external", {name: "@tag", as: "@mytag"}, "@extends", "@shareable", "@inaccessible", "@override"]) {\n          query: Query\n        }\n\n        type Query {\n          _service: _Service!\n          hello: String!\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_adds_link_directive_automatically():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.federation.type(keys=['id'])
    class User:
        id: strawberry.ID

    @strawberry.type
    class Query:
        user: User
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@key"]) {\n          query: Query\n        }\n\n        type Query {\n          _entities(representations: [_Any!]!): [_Entity]!\n          _service: _Service!\n          user: User!\n        }\n\n        type User @key(fields: "id") {\n          id: ID!\n        }\n\n        scalar _Any\n\n        union _Entity = User\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_adds_link_directive_from_interface():
    if False:
        i = 10
        return i + 15

    @strawberry.federation.interface(keys=['id'])
    class SomeInterface:
        id: strawberry.ID

    @strawberry.type
    class User:
        id: strawberry.ID

    @strawberry.type
    class Query:
        user: User
    schema = strawberry.federation.Schema(query=Query, types=[SomeInterface], enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@key"]) {\n          query: Query\n        }\n\n        type Query {\n          _service: _Service!\n          user: User!\n        }\n\n        interface SomeInterface @key(fields: "id") {\n          id: ID!\n        }\n\n        type User {\n          id: ID!\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_adds_link_directive_from_input_types():
    if False:
        print('Hello World!')

    @strawberry.federation.input(inaccessible=True)
    class SomeInput:
        id: strawberry.ID

    @strawberry.type
    class User:
        id: strawberry.ID

    @strawberry.type
    class Query:
        user: User
    schema = strawberry.federation.Schema(query=Query, types=[SomeInput], enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@inaccessible"]) {\n          query: Query\n        }\n\n        type Query {\n          _service: _Service!\n          user: User!\n        }\n\n        input SomeInput @inaccessible {\n          id: ID!\n        }\n\n        type User {\n          id: ID!\n        }\n\n        scalar _Any\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_adds_link_directive_automatically_from_field():
    if False:
        i = 10
        return i + 15

    @strawberry.federation.type(keys=['id'])
    class User:
        id: strawberry.ID
        age: int = strawberry.federation.field(tags=['private'])

    @strawberry.type
    class Query:
        user: User
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@key", "@tag"]) {\n          query: Query\n        }\n\n        type Query {\n          _entities(representations: [_Any!]!): [_Entity]!\n          _service: _Service!\n          user: User!\n        }\n\n        type User @key(fields: "id") {\n          id: ID!\n          age: Int! @tag(name: "private")\n        }\n\n        scalar _Any\n\n        union _Entity = User\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_does_not_add_directive_link_if_federation_two_is_not_enabled():
    if False:
        i = 10
        return i + 15

    @strawberry.federation.type(keys=['id'])
    class User:
        id: strawberry.ID

    @strawberry.type
    class Query:
        user: User
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=False)
    expected = '\n        type Query {\n          _entities(representations: [_Any!]!): [_Entity]!\n          _service: _Service!\n          user: User!\n        }\n\n        type User @key(fields: "id") {\n          id: ID!\n        }\n\n        scalar _Any\n\n        union _Entity = User\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()

def test_adds_link_directive_automatically_from_scalar():
    if False:
        i = 10
        return i + 15

    @strawberry.scalar
    class X:
        pass

    @strawberry.federation.type(keys=['id'])
    class User:
        id: strawberry.ID
        age: X

    @strawberry.type
    class Query:
        user: User
    schema = strawberry.federation.Schema(query=Query, enable_federation_2=True)
    expected = '\n        schema @link(url: "https://specs.apollo.dev/federation/v2.3", import: ["@key"]) {\n          query: Query\n        }\n\n        type Query {\n          _entities(representations: [_Any!]!): [_Entity]!\n          _service: _Service!\n          user: User!\n        }\n\n        type User @key(fields: "id") {\n          id: ID!\n          age: X!\n        }\n\n        scalar X\n\n        scalar _Any\n\n        union _Entity = User\n\n        type _Service {\n          sdl: String!\n        }\n    '
    assert schema.as_str() == textwrap.dedent(expected).strip()