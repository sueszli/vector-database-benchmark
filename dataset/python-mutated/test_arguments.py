import textwrap
from textwrap import dedent
from typing import Optional
from typing_extensions import Annotated
import strawberry
from strawberry.unset import UNSET

def test_argument_descriptions():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Query:

        @strawberry.field
        def hello(name: Annotated[str, strawberry.argument(description='Your name')]='Patrick') -> str:
            if False:
                i = 10
                return i + 15
            return f'Hi {name}'
    schema = strawberry.Schema(query=Query)
    assert str(schema) == dedent('        type Query {\n          hello(\n            """Your name"""\n            name: String! = "Patrick"\n          ): String!\n        }')

def test_argument_deprecation_reason():
    if False:
        print('Hello World!')

    @strawberry.type
    class Query:

        @strawberry.field
        def hello(name: Annotated[str, strawberry.argument(deprecation_reason='Your reason')]='Patrick') -> str:
            if False:
                i = 10
                return i + 15
            return f'Hi {name}'
    schema = strawberry.Schema(query=Query)
    assert str(schema) == dedent('        type Query {\n          hello(name: String! = "Patrick" @deprecated(reason: "Your reason")): String!\n        }')

def test_argument_names():
    if False:
        return 10

    @strawberry.input
    class HelloInput:
        name: str = strawberry.field(default='Patrick', description='Your name')

    @strawberry.type
    class Query:

        @strawberry.field
        def hello(self, input_: Annotated[HelloInput, strawberry.argument(name='input')]) -> str:
            if False:
                print('Hello World!')
            return f'Hi {input_.name}'
    schema = strawberry.Schema(query=Query)
    assert str(schema) == dedent('        input HelloInput {\n          """Your name"""\n          name: String! = "Patrick"\n        }\n\n        type Query {\n          hello(input: HelloInput!): String!\n        }')

def test_argument_with_default_value_none():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Query:

        @strawberry.field
        def hello(self, name: Optional[str]=None) -> str:
            if False:
                print('Hello World!')
            return f'Hi {name}'
    schema = strawberry.Schema(query=Query)
    assert str(schema) == dedent('        type Query {\n          hello(name: String = null): String!\n        }')

def test_optional_argument_unset():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Query:

        @strawberry.field
        def hello(self, name: Optional[str]=UNSET, age: Optional[int]=UNSET) -> str:
            if False:
                i = 10
                return i + 15
            if name is UNSET:
                return 'Hi there'
            return f'Hi {name}'
    schema = strawberry.Schema(query=Query)
    assert str(schema) == dedent('        type Query {\n          hello(name: String, age: Int): String!\n        }')
    result = schema.execute_sync('\n        query {\n            hello\n        }\n    ')
    assert not result.errors
    assert result.data == {'hello': 'Hi there'}

def test_optional_input_field_unset():
    if False:
        print('Hello World!')

    @strawberry.input
    class TestInput:
        name: Optional[str] = UNSET
        age: Optional[int] = UNSET

    @strawberry.type
    class Query:

        @strawberry.field
        def hello(self, input: TestInput) -> str:
            if False:
                print('Hello World!')
            if input.name is UNSET:
                return 'Hi there'
            return f'Hi {input.name}'
    schema = strawberry.Schema(query=Query)
    assert str(schema) == dedent('\n        type Query {\n          hello(input: TestInput!): String!\n        }\n\n        input TestInput {\n          name: String\n          age: Int\n        }\n        ').strip()
    result = schema.execute_sync('\n        query {\n            hello(input: {})\n        }\n    ')
    assert not result.errors
    assert result.data == {'hello': 'Hi there'}

def test_setting_metadata_on_argument():
    if False:
        i = 10
        return i + 15
    field_definition = None

    @strawberry.type
    class Query:

        @strawberry.field
        def hello(self, info, input: Annotated[str, strawberry.argument(metadata={'test': 'foo'})]) -> str:
            if False:
                return 10
            nonlocal field_definition
            field_definition = info._field
            return f'Hi {input}'
    schema = strawberry.Schema(query=Query)
    result = schema.execute_sync('\n        query {\n            hello(input: "there")\n        }\n    ')
    assert not result.errors
    assert result.data == {'hello': 'Hi there'}
    assert field_definition
    assert field_definition.arguments[0].metadata == {'test': 'foo'}

def test_argument_parse_order():
    if False:
        print('Hello World!')
    'Check early early exit from argument parsing due to finding ``info``.\n\n    Reserved argument parsing, which interally also resolves annotations, exits early\n    after detecting the ``info`` argumnent. As a result, the annotation of the ``id_``\n    argument in `tests.schema.test_annotated.type_a.Query` is never resolved. This\n    results in `StrawberryArgument` not being able to detect that ``id_`` makes use of\n    `typing.Annotated` and `strawberry.argument`.\n\n    This behavior is fixed by by ensuring that `StrawberryArgument` makes use of the new\n    `StrawberryAnnotation.evaluate` method instead of consuming the raw annotation.\n\n    An added benefit of this fix is that by removing annotation resolving code from\n    `StrawberryResolver` and making it a part of `StrawberryAnnotation`, it makes it\n    possible for `StrawberryArgument` and `StrawberryResolver` to share the same type\n    evaluation cache.\n\n    Refer to: https://github.com/strawberry-graphql/strawberry/issues/2855\n    '
    from tests.schema.test_annotated import type_a, type_b
    expected = '\n    type Query {\n      getTesting(id: UUID!): String\n    }\n\n    scalar UUID\n    '
    schema_a = strawberry.Schema(type_a.Query)
    schema_b = strawberry.Schema(type_b.Query)
    assert str(schema_a) == str(schema_b)
    assert str(schema_a) == textwrap.dedent(expected).strip()