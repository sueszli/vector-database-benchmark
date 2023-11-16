import re
from pytest import raises
from graphql import parse, get_introspection_query, validate
from ...types import Schema, ObjectType, Interface
from ...types import String, Int, List, Field
from ..depth_limit import depth_limit_validator

class PetType(Interface):
    name = String(required=True)

    class meta:
        name = 'Pet'

class CatType(ObjectType):

    class meta:
        name = 'Cat'
        interfaces = (PetType,)

class DogType(ObjectType):

    class meta:
        name = 'Dog'
        interfaces = (PetType,)

class AddressType(ObjectType):
    street = String(required=True)
    number = Int(required=True)
    city = String(required=True)
    country = String(required=True)

    class Meta:
        name = 'Address'

class HumanType(ObjectType):
    name = String(required=True)
    email = String(required=True)
    address = Field(AddressType, required=True)
    pets = List(PetType, required=True)

    class Meta:
        name = 'Human'

class Query(ObjectType):
    user = Field(HumanType, required=True, name=String())
    version = String(required=True)
    user1 = Field(HumanType, required=True)
    user2 = Field(HumanType, required=True)
    user3 = Field(HumanType, required=True)

    @staticmethod
    def resolve_user(root, info, name=None):
        if False:
            print('Hello World!')
        pass
schema = Schema(query=Query)

def run_query(query: str, max_depth: int, ignore=None):
    if False:
        for i in range(10):
            print('nop')
    document = parse(query)
    result = None

    def callback(query_depths):
        if False:
            return 10
        nonlocal result
        result = query_depths
    errors = validate(schema=schema.graphql_schema, document_ast=document, rules=(depth_limit_validator(max_depth=max_depth, ignore=ignore, callback=callback),))
    return (errors, result)

def test_should_count_depth_without_fragment():
    if False:
        for i in range(10):
            print('nop')
    query = '\n    query read0 {\n      version\n    }\n    query read1 {\n      version\n      user {\n        name\n      }\n    }\n    query read2 {\n      matt: user(name: "matt") {\n        email\n      }\n      andy: user(name: "andy") {\n        email\n        address {\n          city\n        }\n      }\n    }\n    query read3 {\n      matt: user(name: "matt") {\n        email\n      }\n      andy: user(name: "andy") {\n        email\n        address {\n          city\n        }\n        pets {\n          name\n          owner {\n            name\n          }\n        }\n      }\n    }\n    '
    expected = {'read0': 0, 'read1': 1, 'read2': 2, 'read3': 3}
    (errors, result) = run_query(query, 10)
    assert not errors
    assert result == expected

def test_should_count_with_fragments():
    if False:
        return 10
    query = '\n    query read0 {\n      ... on Query {\n        version\n      }\n    }\n    query read1 {\n      version\n      user {\n        ... on Human {\n          name\n        }\n      }\n    }\n    fragment humanInfo on Human {\n      email\n    }\n    fragment petInfo on Pet {\n      name\n      owner {\n        name\n      }\n    }\n    query read2 {\n      matt: user(name: "matt") {\n        ...humanInfo\n      }\n      andy: user(name: "andy") {\n        ...humanInfo\n        address {\n          city\n        }\n      }\n    }\n    query read3 {\n      matt: user(name: "matt") {\n        ...humanInfo\n      }\n      andy: user(name: "andy") {\n        ... on Human {\n          email\n        }\n        address {\n          city\n        }\n        pets {\n          ...petInfo\n        }\n      }\n    }\n  '
    expected = {'read0': 0, 'read1': 1, 'read2': 2, 'read3': 3}
    (errors, result) = run_query(query, 10)
    assert not errors
    assert result == expected

def test_should_ignore_the_introspection_query():
    if False:
        while True:
            i = 10
    (errors, result) = run_query(get_introspection_query(), 10)
    assert not errors
    assert result == {'IntrospectionQuery': 0}

def test_should_catch_very_deep_query():
    if False:
        return 10
    query = '{\n    user {\n      pets {\n        owner {\n          pets {\n            owner {\n              pets {\n                name\n              }\n            }\n          }\n        }\n      }\n    }\n    }\n    '
    (errors, result) = run_query(query, 4)
    assert len(errors) == 1
    assert errors[0].message == "'anonymous' exceeds maximum operation depth of 4."

def test_should_ignore_field():
    if False:
        print('Hello World!')
    query = '\n    query read1 {\n      user { address { city } }\n    }\n    query read2 {\n      user1 { address { city } }\n      user2 { address { city } }\n      user3 { address { city } }\n    }\n    '
    (errors, result) = run_query(query, 10, ignore=['user1', re.compile('user2'), lambda field_name: field_name == 'user3'])
    expected = {'read1': 2, 'read2': 0}
    assert not errors
    assert result == expected

def test_should_raise_invalid_ignore():
    if False:
        while True:
            i = 10
    query = '\n    query read1 {\n      user { address { city } }\n    }\n    '
    with raises(ValueError, match='Invalid ignore option:'):
        run_query(query, 10, ignore=[True])