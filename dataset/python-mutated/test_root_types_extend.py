import textwrap
from strawberry.schema_codegen import codegen

def test_extend_query():
    if False:
        i = 10
        return i + 15
    schema = '\n    extend type Query {\n        world: String!\n    }\n    '
    expected = textwrap.dedent('\n        import strawberry\n\n        @strawberry.type\n        class Query:\n            world: str\n\n        schema = strawberry.Schema(query=Query)\n        ').strip()
    assert codegen(schema).strip() == expected

def test_extend_mutation():
    if False:
        while True:
            i = 10
    schema = '\n    extend type Mutation {\n        world: String!\n    }\n    '
    expected = textwrap.dedent('\n        import strawberry\n\n        @strawberry.type\n        class Mutation:\n            world: str\n\n        schema = strawberry.Schema(mutation=Mutation)\n        ').strip()
    assert codegen(schema).strip() == expected

def test_extend_subscription():
    if False:
        print('Hello World!')
    schema = '\n    extend type Subscription {\n        world: String!\n    }\n    '
    expected = textwrap.dedent('\n        import strawberry\n\n        @strawberry.type\n        class Subscription:\n            world: str\n\n        schema = strawberry.Schema(subscription=Subscription)\n        ').strip()
    assert codegen(schema).strip() == expected