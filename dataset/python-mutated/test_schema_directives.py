import textwrap
from enum import Enum
from typing import List, Optional, Union
from typing_extensions import Annotated
import strawberry
from strawberry.printer import print_schema
from strawberry.schema.config import StrawberryConfig
from strawberry.schema_directive import Location
from strawberry.unset import UNSET

def test_print_simple_directive():
    if False:
        i = 10
        return i + 15

    @strawberry.schema_directive(locations=[Location.FIELD_DEFINITION])
    class Sensitive:
        reason: str

    @strawberry.type
    class Query:
        first_name: str = strawberry.field(directives=[Sensitive(reason='GDPR')])
    expected_output = '\n    directive @sensitive(reason: String!) on FIELD_DEFINITION\n\n    type Query {\n      firstName: String! @sensitive(reason: "GDPR")\n    }\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_print_directive_with_name():
    if False:
        while True:
            i = 10

    @strawberry.schema_directive(name='sensitive', locations=[Location.FIELD_DEFINITION])
    class SensitiveDirective:
        reason: str

    @strawberry.type
    class Query:
        first_name: str = strawberry.field(directives=[SensitiveDirective(reason='GDPR')])
    expected_output = '\n    directive @sensitive(reason: String!) on FIELD_DEFINITION\n\n    type Query {\n      firstName: String! @sensitive(reason: "GDPR")\n    }\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_directive_on_types():
    if False:
        return 10

    @strawberry.input
    class SensitiveValue:
        key: str
        value: str

    @strawberry.schema_directive(locations=[Location.OBJECT, Location.FIELD_DEFINITION])
    class SensitiveData:
        reason: str
        meta: Optional[List[SensitiveValue]] = UNSET

    @strawberry.schema_directive(locations=[Location.INPUT_OBJECT])
    class SensitiveInput:
        reason: str
        meta: Optional[List[SensitiveValue]] = UNSET

    @strawberry.schema_directive(locations=[Location.INPUT_FIELD_DEFINITION])
    class RangeInput:
        min: int
        max: int

    @strawberry.input(directives=[SensitiveInput(reason='GDPR')])
    class Input:
        first_name: str
        age: int = strawberry.field(directives=[RangeInput(min=1, max=100)])

    @strawberry.type(directives=[SensitiveData(reason='GDPR')])
    class User:
        first_name: str
        age: int
        phone: str = strawberry.field(directives=[SensitiveData(reason='PRIVATE', meta=[SensitiveValue(key='can_share_field', value='phone_share_accepted')])])
        phone_share_accepted: bool

    @strawberry.type
    class Query:

        @strawberry.field
        def user(self, input: Input) -> User:
            if False:
                i = 10
                return i + 15
            return User(first_name=input.first_name, age=input.age, phone='+551191551234', phone_share_accepted=False)
    expected_output = '\n    directive @rangeInput(min: Int!, max: Int!) on INPUT_FIELD_DEFINITION\n\n    directive @sensitiveData(reason: String!, meta: [SensitiveValue!]) on OBJECT | FIELD_DEFINITION\n\n    directive @sensitiveInput(reason: String!, meta: [SensitiveValue!]) on INPUT_OBJECT\n\n    input Input @sensitiveInput(reason: "GDPR") {\n      firstName: String!\n      age: Int! @rangeInput(min: 1, max: 100)\n    }\n\n    type Query {\n      user(input: Input!): User!\n    }\n\n    type User @sensitiveData(reason: "GDPR") {\n      firstName: String!\n      age: Int!\n      phone: String! @sensitiveData(reason: "PRIVATE", meta: [{key: "can_share_field", value: "phone_share_accepted"}])\n      phoneShareAccepted: Boolean!\n    }\n\n    input SensitiveValue {\n      key: String!\n      value: String!\n    }\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_using_different_names_for_directive_field():
    if False:
        i = 10
        return i + 15

    @strawberry.schema_directive(locations=[Location.FIELD_DEFINITION])
    class Sensitive:
        reason: str = strawberry.directive_field(name='as')
        real_age: str
        real_age_2: str = strawberry.directive_field(name='real_age')

    @strawberry.type
    class Query:
        first_name: str = strawberry.field(directives=[Sensitive(reason='GDPR', real_age='1', real_age_2='2')])
    expected_output = '\n    directive @sensitive(as: String!, realAge: String!, real_age: String!) on FIELD_DEFINITION\n\n    type Query {\n      firstName: String! @sensitive(as: "GDPR", realAge: "1", real_age: "2")\n    }\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_respects_schema_config_for_names():
    if False:
        while True:
            i = 10

    @strawberry.schema_directive(locations=[Location.FIELD_DEFINITION])
    class Sensitive:
        real_age: str

    @strawberry.type
    class Query:
        first_name: str = strawberry.field(directives=[Sensitive(real_age='42')])
    expected_output = '\n    directive @Sensitive(real_age: String!) on FIELD_DEFINITION\n\n    type Query {\n      first_name: String! @Sensitive(real_age: "42")\n    }\n    '
    schema = strawberry.Schema(query=Query, config=StrawberryConfig(auto_camel_case=False))
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_respects_schema_parameter_types_for_arguments_int():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.schema_directive(locations=[Location.FIELD_DEFINITION])
    class Sensitive:
        real_age: int

    @strawberry.type
    class Query:
        first_name: str = strawberry.field(directives=[Sensitive(real_age=42)])
    expected_output = '\n    directive @Sensitive(real_age: Int!) on FIELD_DEFINITION\n\n    type Query {\n      first_name: String! @Sensitive(real_age: 42)\n    }\n    '
    schema = strawberry.Schema(query=Query, config=StrawberryConfig(auto_camel_case=False))
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_respects_schema_parameter_types_for_arguments_list_of_ints():
    if False:
        return 10

    @strawberry.schema_directive(locations=[Location.FIELD_DEFINITION])
    class Sensitive:
        real_age: List[int]

    @strawberry.type
    class Query:
        first_name: str = strawberry.field(directives=[Sensitive(real_age=[42])])
    expected_output = '\n    directive @Sensitive(real_age: [Int!]!) on FIELD_DEFINITION\n\n    type Query {\n      first_name: String! @Sensitive(real_age: [42])\n    }\n    '
    schema = strawberry.Schema(query=Query, config=StrawberryConfig(auto_camel_case=False))
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_respects_schema_parameter_types_for_arguments_list_of_strings():
    if False:
        while True:
            i = 10

    @strawberry.schema_directive(locations=[Location.FIELD_DEFINITION])
    class Sensitive:
        real_age: List[str]

    @strawberry.type
    class Query:
        first_name: str = strawberry.field(directives=[Sensitive(real_age=['42'])])
    expected_output = '\n    directive @Sensitive(real_age: [String!]!) on FIELD_DEFINITION\n\n    type Query {\n      first_name: String! @Sensitive(real_age: ["42"])\n    }\n    '
    schema = strawberry.Schema(query=Query, config=StrawberryConfig(auto_camel_case=False))
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_prints_directive_on_schema():
    if False:
        i = 10
        return i + 15

    @strawberry.schema_directive(locations=[Location.SCHEMA])
    class Tag:
        name: str

    @strawberry.type
    class Query:
        first_name: str = strawberry.field(directives=[Tag(name='team-1')])
    schema = strawberry.Schema(query=Query, schema_directives=[Tag(name='team-1')])
    expected_output = '\n    directive @tag(name: String!) on SCHEMA\n\n    schema @tag(name: "team-1") {\n      query: Query\n    }\n\n    type Query {\n      firstName: String!\n    }\n    '
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_prints_multiple_directives_on_schema():
    if False:
        while True:
            i = 10

    @strawberry.schema_directive(locations=[Location.SCHEMA])
    class Tag:
        name: str

    @strawberry.type
    class Query:
        first_name: str
    schema = strawberry.Schema(query=Query, schema_directives=[Tag(name='team-1'), Tag(name='team-2')])
    expected_output = '\n    directive @tag(name: String!) on SCHEMA\n\n    schema @tag(name: "team-1") @tag(name: "team-2") {\n      query: Query\n    }\n\n    type Query {\n      firstName: String!\n    }\n    '
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_prints_with_types():
    if False:
        return 10

    @strawberry.input
    class SensitiveConfiguration:
        reason: str

    @strawberry.schema_directive(locations=[Location.FIELD_DEFINITION])
    class Sensitive:
        config: SensitiveConfiguration

    @strawberry.type
    class Query:
        first_name: str = strawberry.field(directives=[Sensitive(config=SensitiveConfiguration(reason='example'))])
    expected_output = '\n    directive @sensitive(config: SensitiveConfiguration!) on FIELD_DEFINITION\n\n    type Query {\n      firstName: String! @sensitive(config: {reason: "example"})\n    }\n\n    input SensitiveConfiguration {\n      reason: String!\n    }\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_prints_with_scalar():
    if False:
        return 10
    SensitiveConfiguration = strawberry.scalar(str, name='SensitiveConfiguration')

    @strawberry.schema_directive(locations=[Location.FIELD_DEFINITION])
    class Sensitive:
        config: SensitiveConfiguration

    @strawberry.type
    class Query:
        first_name: str = strawberry.field(directives=[Sensitive(config='Some config')])
    expected_output = '\n    directive @sensitive(config: SensitiveConfiguration!) on FIELD_DEFINITION\n\n    type Query {\n      firstName: String! @sensitive(config: "Some config")\n    }\n\n    scalar SensitiveConfiguration\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_prints_with_enum():
    if False:
        print('Hello World!')

    @strawberry.enum
    class Reason(str, Enum):
        EXAMPLE = 'example'
        __slots__ = ()

    @strawberry.schema_directive(locations=[Location.FIELD_DEFINITION])
    class Sensitive:
        reason: Reason

    @strawberry.type
    class Query:
        first_name: str = strawberry.field(directives=[Sensitive(reason=Reason.EXAMPLE)])
    expected_output = '\n    directive @sensitive(reason: Reason!) on FIELD_DEFINITION\n\n    type Query {\n      firstName: String! @sensitive(reason: EXAMPLE)\n    }\n\n    enum Reason {\n      EXAMPLE\n    }\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_does_not_print_definition():
    if False:
        print('Hello World!')

    @strawberry.schema_directive(locations=[Location.FIELD_DEFINITION], print_definition=False)
    class Sensitive:
        reason: str

    @strawberry.type
    class Query:
        first_name: str = strawberry.field(directives=[Sensitive(reason='GDPR')])
    expected_output = '\n    type Query {\n      firstName: String! @sensitive(reason: "GDPR")\n    }\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_print_directive_on_scalar():
    if False:
        return 10

    @strawberry.schema_directive(locations=[Location.SCALAR])
    class Sensitive:
        reason: str
    SensitiveString = strawberry.scalar(str, name='SensitiveString', directives=[Sensitive(reason='example')])

    @strawberry.type
    class Query:
        first_name: SensitiveString
    expected_output = '\n    directive @sensitive(reason: String!) on SCALAR\n\n    type Query {\n      firstName: SensitiveString!\n    }\n\n    scalar SensitiveString @sensitive(reason: "example")\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_print_directive_on_enum():
    if False:
        i = 10
        return i + 15

    @strawberry.schema_directive(locations=[Location.ENUM])
    class Sensitive:
        reason: str

    @strawberry.enum(directives=[Sensitive(reason='example')])
    class SomeEnum(str, Enum):
        EXAMPLE = 'example'
        __slots__ = ()

    @strawberry.type
    class Query:
        first_name: SomeEnum
    expected_output = '\n    directive @sensitive(reason: String!) on ENUM\n\n    type Query {\n      firstName: SomeEnum!\n    }\n\n    enum SomeEnum @sensitive(reason: "example") {\n      EXAMPLE\n    }\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_print_directive_on_enum_value():
    if False:
        while True:
            i = 10

    @strawberry.schema_directive(locations=[Location.ENUM_VALUE])
    class Sensitive:
        reason: str

    @strawberry.enum
    class SomeEnum(Enum):
        EXAMPLE = strawberry.enum_value('example', directives=[Sensitive(reason='example')])

    @strawberry.type
    class Query:
        first_name: SomeEnum
    expected_output = '\n    directive @sensitive(reason: String!) on ENUM_VALUE\n\n    type Query {\n      firstName: SomeEnum!\n    }\n\n    enum SomeEnum {\n      EXAMPLE @sensitive(reason: "example")\n    }\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_print_directive_on_union():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class A:
        a: int

    @strawberry.type
    class B:
        b: int

    @strawberry.schema_directive(locations=[Location.SCALAR])
    class Sensitive:
        reason: str
    MyUnion = Annotated[Union[A, B], strawberry.union(name='MyUnion', directives=[Sensitive(reason='example')])]

    @strawberry.type
    class Query:
        example: MyUnion
    expected_output = '\n    directive @sensitive(reason: String!) on SCALAR\n\n    type A {\n      a: Int!\n    }\n\n    type B {\n      b: Int!\n    }\n\n    union MyUnion @sensitive(reason: "example") = A | B\n\n    type Query {\n      example: MyUnion!\n    }\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_print_directive_on_argument():
    if False:
        return 10

    @strawberry.schema_directive(locations=[Location.ARGUMENT_DEFINITION])
    class Sensitive:
        reason: str

    @strawberry.type
    class Query:

        @strawberry.field
        def hello(self, name: Annotated[str, strawberry.argument(directives=[Sensitive(reason='example')])], age: Annotated[str, strawberry.argument(directives=[Sensitive(reason='example')])]) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return f'Hello {name} of {age}'
    expected_output = '\n    directive @sensitive(reason: String!) on ARGUMENT_DEFINITION\n\n    type Query {\n      hello(name: String! @sensitive(reason: "example"), age: String! @sensitive(reason: "example")): String!\n    }\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()

def test_print_directive_on_argument_with_description():
    if False:
        i = 10
        return i + 15

    @strawberry.schema_directive(locations=[Location.ARGUMENT_DEFINITION])
    class Sensitive:
        reason: str

    @strawberry.type
    class Query:

        @strawberry.field
        def hello(self, name: Annotated[str, strawberry.argument(description='Name', directives=[Sensitive(reason='example')])], age: Annotated[str, strawberry.argument(directives=[Sensitive(reason='example')])]) -> str:
            if False:
                print('Hello World!')
            return f'Hello {name} of {age}'
    expected_output = '\n    directive @sensitive(reason: String!) on ARGUMENT_DEFINITION\n\n    type Query {\n      hello(\n        """Name"""\n        name: String! @sensitive(reason: "example")\n        age: String! @sensitive(reason: "example")\n      ): String!\n    }\n    '
    schema = strawberry.Schema(query=Query)
    assert print_schema(schema) == textwrap.dedent(expected_output).strip()