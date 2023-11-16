from textwrap import dedent
import pytest
import strawberry
from strawberry.annotation import StrawberryAnnotation
from strawberry.field import StrawberryField
from strawberry.tools import create_type
from strawberry.type import get_object_definition

def test_create_type():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.field
    def name() -> str:
        if False:
            while True:
                i = 10
        return 'foo'
    MyType = create_type('MyType', [name], description='This is a description')
    definition = get_object_definition(MyType, strict=True)
    assert definition.name == 'MyType'
    assert definition.description == 'This is a description'
    assert definition.is_input is False
    assert len(definition.fields) == 1
    assert definition.fields[0].python_name == 'name'
    assert definition.fields[0].graphql_name is None
    assert definition.fields[0].type == str

def test_create_type_extend_and_directives():
    if False:
        i = 10
        return i + 15

    @strawberry.field
    def name() -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'foo'
    MyType = create_type('MyType', [name], description='This is a description', extend=True, directives=[object()])
    definition = get_object_definition(MyType, strict=True)
    assert definition.name == 'MyType'
    assert definition.description == 'This is a description'
    assert definition.is_input is False
    assert definition.extend is True
    assert len(list(definition.directives)) == 1
    assert len(definition.fields) == 1
    assert definition.fields[0].python_name == 'name'
    assert definition.fields[0].graphql_name is None
    assert definition.fields[0].type == str

def test_create_input_type():
    if False:
        return 10
    name = StrawberryField(python_name='name', type_annotation=StrawberryAnnotation(str))
    MyType = create_type('MyType', [name], is_input=True, description='This is a description')
    definition = get_object_definition(MyType, strict=True)
    assert definition.name == 'MyType'
    assert definition.description == 'This is a description'
    assert definition.is_input
    assert len(definition.fields) == 1
    assert definition.fields[0].python_name == 'name'
    assert definition.fields[0].graphql_name is None
    assert definition.fields[0].type == str

def test_create_interface_type():
    if False:
        print('Hello World!')
    name = StrawberryField(python_name='name', type_annotation=StrawberryAnnotation(str))
    MyType = create_type('MyType', [name], is_interface=True, description='This is a description')
    definition = get_object_definition(MyType, strict=True)
    assert definition.name == 'MyType'
    assert definition.description == 'This is a description'
    assert definition.is_input is False
    assert definition.is_interface
    assert len(definition.fields) == 1
    assert definition.fields[0].python_name == 'name'
    assert definition.fields[0].graphql_name is None
    assert definition.fields[0].type == str

def test_create_variable_type():
    if False:
        print('Hello World!')

    def get_name() -> str:
        if False:
            return 10
        return 'foo'
    name = strawberry.field(name='name', resolver=get_name)
    MyType = create_type('MyType', [name])
    definition = get_object_definition(MyType, strict=True)
    assert len(definition.fields) == 1
    assert definition.fields[0].python_name == 'get_name'
    assert definition.fields[0].graphql_name == 'name'
    assert definition.fields[0].type == str

def test_create_type_empty_list():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        create_type('MyType', [])

def test_create_type_field_no_name():
    if False:
        for i in range(10):
            print('nop')
    name = strawberry.field()
    with pytest.raises(ValueError):
        create_type('MyType', [name])

def test_create_type_field_invalid():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError):
        create_type('MyType', [strawberry.type()])

def test_create_mutation_type():
    if False:
        print('Hello World!')

    @strawberry.type
    class User:
        username: str

    @strawberry.mutation
    def make_user(info, username: str) -> User:
        if False:
            print('Hello World!')
        return User(username=username)
    Mutation = create_type('Mutation', [make_user])
    definition = get_object_definition(Mutation, strict=True)
    assert len(definition.fields) == 1
    assert definition.fields[0].python_name == 'make_user'
    assert definition.fields[0].graphql_name is None
    assert definition.fields[0].type == User

def test_create_mutation_type_with_params():
    if False:
        print('Hello World!')

    @strawberry.type
    class User:
        username: str

    @strawberry.mutation(name='makeNewUser', description='Make a new user')
    def make_user(info, username: str) -> User:
        if False:
            i = 10
            return i + 15
        return User(username=username)
    Mutation = create_type('Mutation', [make_user])
    definition = get_object_definition(Mutation, strict=True)
    assert len(definition.fields) == 1
    assert definition.fields[0].python_name == 'make_user'
    assert definition.fields[0].graphql_name == 'makeNewUser'
    assert definition.fields[0].type == User
    assert definition.fields[0].description == 'Make a new user'

def test_create_schema():
    if False:
        print('Hello World!')

    @strawberry.type
    class User:
        id: strawberry.ID

    @strawberry.field
    def get_user_by_id(info, id: strawberry.ID) -> User:
        if False:
            for i in range(10):
                print('nop')
        return User(id=id)
    Query = create_type('Query', [get_user_by_id])
    schema = strawberry.Schema(query=Query)
    sdl = '\n    type Query {\n      getUserById(id: ID!): User!\n    }\n\n    type User {\n      id: ID!\n    }\n    '
    assert dedent(sdl).strip() == str(schema)
    result = schema.execute_sync('\n        {\n            getUserById(id: "TEST") {\n                id\n            }\n        }\n    ')
    assert not result.errors
    assert result.data == {'getUserById': {'id': 'TEST'}}