from textwrap import dedent
import pytest
import strawberry
from strawberry.tools import merge_types

@strawberry.type
class Person:

    @strawberry.field
    def name(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'Eve'

    @strawberry.field
    def age(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 42

@strawberry.type
class SimpleGreeter:

    @strawberry.field
    def hi(self) -> str:
        if False:
            return 10
        return 'Hi'

@strawberry.type
class ComplexGreeter:

    @strawberry.field
    def hi(self, name: str='world') -> str:
        if False:
            i = 10
            return i + 15
        return f'Hello, {name}!'

    @strawberry.field
    def bye(self, name: str='world') -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'Bye, {name}!'

def test_custom_name():
    if False:
        for i in range(10):
            print('nop')
    'The resulting type should have a custom name is one is specified'
    custom_name = 'SuperQuery'
    ComboQuery = merge_types(custom_name, (ComplexGreeter, Person))
    assert ComboQuery.__name__ == custom_name

def test_inheritance():
    if False:
        i = 10
        return i + 15
    'It should merge multiple types following the regular inheritance rules'
    ComboQuery = merge_types('SuperType', (ComplexGreeter, Person))
    definition = ComboQuery.__strawberry_definition__
    assert len(definition.fields) == 4
    actuals = [(field.python_name, field.type) for field in definition.fields]
    expected = [('hi', str), ('bye', str), ('name', str), ('age', int)]
    assert actuals == expected

def test_empty_list():
    if False:
        print('Hello World!')
    'It should raise when the `types` argument is empty'
    with pytest.raises(ValueError):
        merge_types('EmptyType', ())

def test_schema():
    if False:
        while True:
            i = 10
    'It should create a valid, usable schema based on a merged query'
    ComboQuery = merge_types('SuperSchema', (ComplexGreeter, Person))
    schema = strawberry.Schema(query=ComboQuery)
    sdl = '\n        schema {\n          query: SuperSchema\n        }\n\n        type SuperSchema {\n          hi(name: String! = "world"): String!\n          bye(name: String! = "world"): String!\n          name: String!\n          age: Int!\n        }\n    '
    assert dedent(sdl).strip() == str(schema)
    result = schema.execute_sync('query { hi }')
    assert not result.errors
    assert result.data == {'hi': 'Hello, world!'}

def test_fields_override():
    if False:
        return 10
    'It should warn when merging results in overriding fields'
    with pytest.warns(Warning):
        merge_types('FieldsOverride', (ComplexGreeter, SimpleGreeter))