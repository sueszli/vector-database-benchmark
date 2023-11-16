import textwrap
from enum import Enum
from typing import Any, Dict, List, NoReturn, Optional
import pytest
import strawberry
from strawberry.directive import DirectiveLocation, DirectiveValue
from strawberry.extensions import SchemaExtension
from strawberry.schema.config import StrawberryConfig
from strawberry.type import get_object_definition
from strawberry.types.info import Info
from strawberry.utils.await_maybe import await_maybe

def test_supports_default_directives():
    if False:
        return 10

    @strawberry.type
    class Person:
        name: str = 'Jess'
        points: int = 2000

    @strawberry.type
    class Query:

        @strawberry.field
        def person(self) -> Person:
            if False:
                while True:
                    i = 10
            return Person()
    query = 'query ($includePoints: Boolean!){\n        person {\n            name\n            points @include(if: $includePoints)\n        }\n    }'
    schema = strawberry.Schema(query=Query)
    result = schema.execute_sync(query, variable_values={'includePoints': False}, context_value={'username': 'foo'})
    assert not result.errors
    assert result.data['person'] == {'name': 'Jess'}
    query = 'query ($skipPoints: Boolean!){\n        person {\n            name\n            points @skip(if: $skipPoints)\n        }\n    }'
    schema = strawberry.Schema(query=Query)
    result = schema.execute_sync(query, variable_values={'skipPoints': False})
    assert not result.errors
    assert result.data['person'] == {'name': 'Jess', 'points': 2000}

@pytest.mark.asyncio
async def test_supports_default_directives_async():

    @strawberry.type
    class Person:
        name: str = 'Jess'
        points: int = 2000

    @strawberry.type
    class Query:

        @strawberry.field
        def person(self) -> Person:
            if False:
                for i in range(10):
                    print('nop')
            return Person()
    query = 'query ($includePoints: Boolean!){\n        person {\n            name\n            points @include(if: $includePoints)\n        }\n    }'
    schema = strawberry.Schema(query=Query)
    result = await schema.execute(query, variable_values={'includePoints': False})
    assert not result.errors
    assert result.data['person'] == {'name': 'Jess'}
    query = 'query ($skipPoints: Boolean!){\n        person {\n            name\n            points @skip(if: $skipPoints)\n        }\n    }'
    schema = strawberry.Schema(query=Query)
    result = await schema.execute(query, variable_values={'skipPoints': False})
    assert not result.errors
    assert result.data['person'] == {'name': 'Jess', 'points': 2000}

def test_can_declare_directives():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Query:
        cake: str = 'made_in_switzerland'

    @strawberry.directive(locations=[DirectiveLocation.FIELD], description='Make string uppercase')
    def uppercase(value: str, example: str):
        if False:
            print('Hello World!')
        return value.upper()
    schema = strawberry.Schema(query=Query, directives=[uppercase])
    expected_schema = '\n    """Make string uppercase"""\n    directive @uppercase(example: String!) on FIELD\n\n    type Query {\n      cake: String!\n    }\n    '
    assert schema.as_str() == textwrap.dedent(expected_schema).strip()

def test_directive_arguments_without_value_param():
    if False:
        i = 10
        return i + 15
    'Regression test for Strawberry Issue #1666.\n\n    https://github.com/strawberry-graphql/strawberry/issues/1666\n    '

    @strawberry.type
    class Query:
        cake: str = 'victoria sponge'

    @strawberry.directive(locations=[DirectiveLocation.FIELD], description="Don't actually like cake? try ice cream instead")
    def ice_cream(flavor: str):
        if False:
            i = 10
            return i + 15
        return f'{flavor} ice cream'
    schema = strawberry.Schema(query=Query, directives=[ice_cream])
    expected_schema = '\n    """Don\'t actually like cake? try ice cream instead"""\n    directive @iceCream(flavor: String!) on FIELD\n\n    type Query {\n      cake: String!\n    }\n    '
    assert schema.as_str() == textwrap.dedent(expected_schema).strip()
    query = 'query { cake @iceCream(flavor: "strawberry") }'
    result = schema.execute_sync(query, root_value=Query())
    assert result.data == {'cake': 'strawberry ice cream'}

def test_runs_directives():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class Person:
        name: str = 'Jess'

    @strawberry.type
    class Query:

        @strawberry.field
        def person(self) -> Person:
            if False:
                return 10
            return Person()

    @strawberry.directive(locations=[DirectiveLocation.FIELD], description='Make string uppercase')
    def turn_uppercase(value: str):
        if False:
            for i in range(10):
                print('nop')
        return value.upper()

    @strawberry.directive(locations=[DirectiveLocation.FIELD])
    def replace(value: str, old: str, new: str):
        if False:
            print('Hello World!')
        return value.replace(old, new)
    schema = strawberry.Schema(query=Query, directives=[turn_uppercase, replace])
    query = 'query People($identified: Boolean!){\n        person {\n            name @turnUppercase\n        }\n        jess: person {\n            name @replace(old: "Jess", new: "Jessica")\n        }\n        johnDoe: person {\n            name @replace(old: "Jess", new: "John") @include(if: $identified)\n        }\n    }'
    result = schema.execute_sync(query, variable_values={'identified': False})
    assert not result.errors
    assert result.data['person']['name'] == 'JESS'
    assert result.data['jess']['name'] == 'Jessica'
    assert result.data['johnDoe'].get('name') is None

def test_runs_directives_camel_case_off():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Person:
        name: str = 'Jess'

    @strawberry.type
    class Query:

        @strawberry.field
        def person(self) -> Person:
            if False:
                print('Hello World!')
            return Person()

    @strawberry.directive(locations=[DirectiveLocation.FIELD], description='Make string uppercase')
    def turn_uppercase(value: str):
        if False:
            return 10
        return value.upper()

    @strawberry.directive(locations=[DirectiveLocation.FIELD])
    def replace(value: str, old: str, new: str):
        if False:
            while True:
                i = 10
        return value.replace(old, new)
    schema = strawberry.Schema(query=Query, directives=[turn_uppercase, replace], config=StrawberryConfig(auto_camel_case=False))
    query = 'query People($identified: Boolean!){\n        person {\n            name @turn_uppercase\n        }\n        jess: person {\n            name @replace(old: "Jess", new: "Jessica")\n        }\n        johnDoe: person {\n            name @replace(old: "Jess", new: "John") @include(if: $identified)\n        }\n    }'
    result = schema.execute_sync(query, variable_values={'identified': False})
    assert not result.errors
    assert result.data['person']['name'] == 'JESS'
    assert result.data['jess']['name'] == 'Jessica'
    assert result.data['johnDoe'].get('name') is None

@pytest.mark.asyncio
async def test_runs_directives_async():

    @strawberry.type
    class Person:
        name: str = 'Jess'

    @strawberry.type
    class Query:

        @strawberry.field
        def person(self) -> Person:
            if False:
                return 10
            return Person()

    @strawberry.directive(locations=[DirectiveLocation.FIELD], description='Make string uppercase')
    async def uppercase(value: str):
        return value.upper()
    schema = strawberry.Schema(query=Query, directives=[uppercase])
    query = '{\n        person {\n            name @uppercase\n        }\n    }'
    result = await schema.execute(query, variable_values={'identified': False})
    assert not result.errors
    assert result.data
    assert result.data['person']['name'] == 'JESS'

@pytest.mark.xfail
def test_runs_directives_with_list_params():
    if False:
        return 10

    @strawberry.type
    class Person:
        name: str = 'Jess'

    @strawberry.type
    class Query:

        @strawberry.field
        def person(self) -> Person:
            if False:
                i = 10
                return i + 15
            return Person()

    @strawberry.directive(locations=[DirectiveLocation.FIELD])
    def replace(value: str, old_list: List[str], new: str):
        if False:
            for i in range(10):
                print('nop')
        for old in old_list:
            value = value.replace(old, new)
        return value
    schema = strawberry.Schema(query=Query, directives=[replace])
    query = 'query People {\n        person {\n            name @replace(oldList: ["J", "e", "s", "s"], new: "John")\n        }\n    }'
    result = schema.execute_sync(query, variable_values={'identified': False})
    assert not result.errors
    assert result.data['person']['name'] == 'JESS'

def test_runs_directives_with_extensions():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Person:
        name: str = 'Jess'

    @strawberry.type
    class Query:

        @strawberry.field
        def person(self) -> Person:
            if False:
                while True:
                    i = 10
            return Person()

    @strawberry.directive(locations=[DirectiveLocation.FIELD], description='Make string uppercase')
    def uppercase(value: str):
        if False:
            print('Hello World!')
        return value.upper()

    class ExampleExtension(SchemaExtension):

        def resolve(self, _next, root, info, *args: str, **kwargs: Any):
            if False:
                return 10
            return _next(root, info, *args, **kwargs)
    schema = strawberry.Schema(query=Query, directives=[uppercase], extensions=[ExampleExtension])
    query = 'query {\n        person {\n            name @uppercase\n        }\n    }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data
    assert result.data['person']['name'] == 'JESS'

@pytest.mark.asyncio
async def test_runs_directives_with_extensions_async():

    @strawberry.type
    class Person:
        name: str = 'Jess'

    @strawberry.type
    class Query:

        @strawberry.field
        async def person(self) -> Person:
            return Person()

    @strawberry.directive(locations=[DirectiveLocation.FIELD], description='Make string uppercase')
    def uppercase(value: str):
        if False:
            i = 10
            return i + 15
        return value.upper()

    class ExampleExtension(SchemaExtension):

        async def resolve(self, _next, root, info, *args: str, **kwargs: Any):
            return await await_maybe(_next(root, info, *args, **kwargs))
    schema = strawberry.Schema(query=Query, directives=[uppercase], extensions=[ExampleExtension])
    query = 'query {\n        person {\n            name @uppercase\n        }\n    }'
    result = await schema.execute(query)
    assert not result.errors
    assert result.data
    assert result.data['person']['name'] == 'JESS'

@pytest.fixture
def info_directive_schema() -> strawberry.Schema:
    if False:
        i = 10
        return i + 15
    'Returns a schema with directive that validates if info is recieved.'

    @strawberry.enum
    class Locale(Enum):
        EN: str = 'EN'
        NL: str = 'NL'
    greetings: Dict[Locale, str] = {Locale.EN: 'Hello {username}', Locale.NL: 'Hallo {username}'}

    @strawberry.type
    class Query:

        @strawberry.field
        def greetingTemplate(self, locale: Locale=Locale.EN) -> str:
            if False:
                return 10
            return greetings[locale]
    field = get_object_definition(Query, strict=True).fields[0]

    @strawberry.directive(locations=[DirectiveLocation.FIELD], description='Interpolate string on the server from context data')
    def interpolate(value: str, info: Info):
        if False:
            i = 10
            return i + 15
        try:
            assert isinstance(info, Info)
            assert info._field is field
            return value.format(**info.context['userdata'])
        except KeyError:
            return value
    return strawberry.Schema(query=Query, directives=[interpolate])

def test_info_directive_schema(info_directive_schema: strawberry.Schema):
    if False:
        for i in range(10):
            print('nop')
    expected_schema = '\n    """Interpolate string on the server from context data"""\n    directive @interpolate on FIELD\n\n    enum Locale {\n      EN\n      NL\n    }\n\n    type Query {\n      greetingTemplate(locale: Locale! = EN): String!\n    }\n    '
    assert textwrap.dedent(expected_schema).strip() == str(info_directive_schema)

def test_info_directive(info_directive_schema: strawberry.Schema):
    if False:
        for i in range(10):
            print('nop')
    query = 'query { greetingTemplate @interpolate }'
    result = info_directive_schema.execute_sync(query, context_value={'userdata': {'username': 'Foo'}})
    assert result.data == {'greetingTemplate': 'Hello Foo'}

@pytest.mark.asyncio
async def test_info_directive_async(info_directive_schema: strawberry.Schema):
    query = 'query { greetingTemplate @interpolate }'
    result = await info_directive_schema.execute(query, context_value={'userdata': {'username': 'Foo'}})
    assert result.data == {'greetingTemplate': 'Hello Foo'}

def test_directive_value():
    if False:
        print('Hello World!')
    'Tests if directive value is detected by type instead of by arg-name `value`.'

    @strawberry.type
    class Cake:
        frosting: Optional[str] = None
        flavor: str = 'Chocolate'

    @strawberry.type
    class Query:

        @strawberry.field
        def cake(self) -> Cake:
            if False:
                i = 10
                return i + 15
            return Cake()

    @strawberry.directive(locations=[DirectiveLocation.FIELD], description='Add frostring with ``flavor`` to a cake.')
    def add_frosting(flavor: str, v: DirectiveValue[Cake], value: str):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(v, Cake)
        assert value == 'foo'
        v.frosting = flavor
        return v
    schema = strawberry.Schema(query=Query, directives=[add_frosting])
    result = schema.execute_sync('query {\n            cake @addFrosting(flavor: "Vanilla", value: "foo") {\n                frosting\n                flavor\n            }\n        }\n        ')
    assert result.data == {'cake': {'frosting': 'Vanilla', 'flavor': 'Chocolate'}}

@strawberry.directive(locations=[DirectiveLocation.FIELD], description='Add frostring with ``flavor`` to a cake.')
def add_frosting(flavor: str, v: DirectiveValue['Cake'], value: str) -> 'Cake':
    if False:
        i = 10
        return i + 15
    assert isinstance(v, Cake)
    assert value == 'foo'
    v.frosting = flavor
    return v

@strawberry.type
class Query:

    @strawberry.field
    def cake(self) -> 'Cake':
        if False:
            while True:
                i = 10
        return Cake()

@strawberry.type
class Cake:
    frosting: Optional[str] = None
    flavor: str = 'Chocolate'

def test_directive_value_forward_ref():
    if False:
        i = 10
        return i + 15
    'Tests if directive value by type works with PEP-563.'
    schema = strawberry.Schema(query=Query, directives=[add_frosting])
    result = schema.execute_sync('query {\n            cake @addFrosting(flavor: "Vanilla", value: "foo") {\n                frosting\n                flavor\n            }\n        }\n        ')
    assert result.data == {'cake': {'frosting': 'Vanilla', 'flavor': 'Chocolate'}}

def test_name_first_directive_value():
    if False:
        return 10

    @strawberry.type
    class Query:

        @strawberry.field
        def greeting(self) -> str:
            if False:
                print('Hello World!')
            return 'Hi'

    @strawberry.directive(locations=[DirectiveLocation.FIELD])
    def personalize_greeting(value: str, v: DirectiveValue[str]):
        if False:
            i = 10
            return i + 15
        assert v == 'Hi'
        return f'{v} {value}'
    schema = strawberry.Schema(Query, directives=[personalize_greeting])
    result = schema.execute_sync('{ greeting @personalizeGreeting(value: "Bar")}')
    assert result.data is not None
    assert not result.errors
    assert result.data['greeting'] == 'Hi Bar'

def test_named_based_directive_value_is_deprecated():
    if False:
        print('Hello World!')
    with pytest.deprecated_call(match="Argument name-based matching of 'value'"):

        @strawberry.type
        class Query:
            hello: str = 'hello'

        @strawberry.directive(locations=[DirectiveLocation.FIELD])
        def deprecated_value(value):
            if False:
                for i in range(10):
                    print('nop')
            ...
        strawberry.Schema(query=Query, directives=[deprecated_value])

@pytest.mark.asyncio
async def test_directive_list_argument() -> NoReturn:

    @strawberry.type
    class Query:

        @strawberry.field
        def greeting(self) -> str:
            if False:
                while True:
                    i = 10
            return 'Hi'

    @strawberry.directive(locations=[DirectiveLocation.FIELD])
    def append_names(value: DirectiveValue[str], names: List[str]):
        if False:
            return 10
        assert isinstance(names, list)
        return f"{value} {', '.join(names)}"
    schema = strawberry.Schema(query=Query, directives=[append_names])
    result = await schema.execute('query { greeting @appendNames(names: ["foo", "bar"])}')
    assert result.errors is None
    assert result.data['greeting'] == 'Hi foo, bar'

def test_directives_with_custom_types():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.input
    class DirectiveInput:
        example: str

    @strawberry.type
    class Query:
        cake: str = 'made_in_switzerland'

    @strawberry.directive(locations=[DirectiveLocation.FIELD], description='Make string uppercase')
    def uppercase(value: str, input: DirectiveInput):
        if False:
            for i in range(10):
                print('nop')
        return value.upper()
    schema = strawberry.Schema(query=Query, directives=[uppercase])
    expected_schema = '\n    """Make string uppercase"""\n    directive @uppercase(input: DirectiveInput!) on FIELD\n\n    input DirectiveInput {\n      example: String!\n    }\n\n    type Query {\n      cake: String!\n    }\n    '
    assert schema.as_str() == textwrap.dedent(expected_schema).strip()

def test_directives_with_scalar():
    if False:
        for i in range(10):
            print('nop')
    DirectiveInput = strawberry.scalar(str, name='DirectiveInput')

    @strawberry.type
    class Query:
        cake: str = 'made_in_switzerland'

    @strawberry.directive(locations=[DirectiveLocation.FIELD], description='Make string uppercase')
    def uppercase(value: str, input: DirectiveInput):
        if False:
            return 10
        return value.upper()
    schema = strawberry.Schema(query=Query, directives=[uppercase])
    expected_schema = '\n    """Make string uppercase"""\n    directive @uppercase(input: DirectiveInput!) on FIELD\n\n    scalar DirectiveInput\n\n    type Query {\n      cake: String!\n    }\n    '
    assert schema.as_str() == textwrap.dedent(expected_schema).strip()