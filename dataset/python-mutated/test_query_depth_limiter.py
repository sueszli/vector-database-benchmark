from typing import Dict, List, Optional, Tuple, Union
import pytest
from graphql import GraphQLError, get_introspection_query, parse, specified_rules, validate
import strawberry
from strawberry.extensions import QueryDepthLimiter
from strawberry.extensions.query_depth_limiter import IgnoreContext, ShouldIgnoreType, create_validator

@strawberry.interface
class Pet:
    name: str
    owner: 'Human'

@strawberry.type
class Cat(Pet):
    pass

@strawberry.type
class Dog(Pet):
    pass

@strawberry.type
class Address:
    street: str
    number: int
    city: str
    country: str

@strawberry.type
class Human:
    name: str
    email: str
    address: Address
    pets: List[Pet]

@strawberry.input
class Biography:
    name: str
    owner_name: str

@strawberry.type
class Query:

    @strawberry.field
    def user(self, name: Optional[str], id: Optional[int], age: Optional[float], is_cool: Optional[bool]) -> Human:
        if False:
            return 10
        pass

    @strawberry.field
    def users(self, names: Optional[List[str]]) -> List[Human]:
        if False:
            return 10
        pass

    @strawberry.field
    def cat(bio: Biography) -> Cat:
        if False:
            for i in range(10):
                print('nop')
        pass
    version: str
    user1: Human
    user2: Human
    user3: Human
schema = strawberry.Schema(Query)

def run_query(query: str, max_depth: int, should_ignore: ShouldIgnoreType=None) -> Tuple[List[GraphQLError], Union[Dict[str, int], None]]:
    if False:
        print('Hello World!')
    document = parse(query)
    result = None

    def callback(query_depths):
        if False:
            while True:
                i = 10
        nonlocal result
        result = query_depths
    validation_rule = create_validator(max_depth, should_ignore, callback)
    errors = validate(schema._schema, document, rules=(*specified_rules, validation_rule))
    return (errors, result)

def test_should_count_depth_without_fragment():
    if False:
        while True:
            i = 10
    query = '\n    query read0 {\n      version\n    }\n    query read1 {\n      version\n      user {\n        name\n      }\n    }\n    query read2 {\n      matt: user(name: "matt") {\n        email\n      }\n      andy: user(name: "andy") {\n        email\n        address {\n          city\n        }\n      }\n    }\n    query read3 {\n      matt: user(name: "matt") {\n        email\n      }\n      andy: user(name: "andy") {\n        email\n        address {\n          city\n        }\n        pets {\n          name\n          owner {\n            name\n          }\n        }\n      }\n    }\n    '
    expected = {'read0': 0, 'read1': 1, 'read2': 2, 'read3': 3}
    (errors, result) = run_query(query, 10)
    assert not errors
    assert result == expected

def test_should_count_with_fragments():
    if False:
        while True:
            i = 10
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

def test_should_catch_query_thats_too_deep():
    if False:
        for i in range(10):
            print('nop')
    query = '{\n    user {\n      pets {\n        owner {\n          pets {\n            owner {\n              pets {\n                name\n              }\n            }\n          }\n        }\n      }\n    }\n    }\n    '
    (errors, result) = run_query(query, 4)
    assert len(errors) == 1
    assert errors[0].message == "'anonymous' exceeds maximum operation depth of 4"

def test_should_raise_invalid_ignore():
    if False:
        print('Hello World!')
    query = '\n    query read1 {\n      user { address { city } }\n    }\n    '
    with pytest.raises(TypeError, match='The `should_ignore` argument to `QueryDepthLimiter` must be a callable.'):
        strawberry.Schema(Query, extensions=[QueryDepthLimiter(max_depth=10, should_ignore=True)])

def test_should_ignore_field_by_name():
    if False:
        print('Hello World!')
    query = '\n    query read1 {\n      user { address { city } }\n    }\n    query read2 {\n      user1 { address { city } }\n      user2 { address { city } }\n      user3 { address { city } }\n    }\n    '

    def should_ignore(ignore: IgnoreContext) -> bool:
        if False:
            print('Hello World!')
        return ignore.field_name == 'user1' or ignore.field_name == 'user2' or ignore.field_name == 'user3'
    (errors, result) = run_query(query, 10, should_ignore=should_ignore)
    expected = {'read1': 2, 'read2': 0}
    assert not errors
    assert result == expected

def test_should_ignore_field_by_str_argument():
    if False:
        while True:
            i = 10
    query = '\n    query read1 {\n      user(name:"matt") { address { city } }\n    }\n    query read2 {\n      user1 { address { city } }\n      user2 { address { city } }\n      user3 { address { city } }\n    }\n    '

    def should_ignore(ignore: IgnoreContext) -> bool:
        if False:
            return 10
        return ignore.field_args.get('name') == 'matt'
    (errors, result) = run_query(query, 10, should_ignore=should_ignore)
    expected = {'read1': 0, 'read2': 2}
    assert not errors
    assert result == expected

def test_should_ignore_field_by_int_argument():
    if False:
        for i in range(10):
            print('nop')
    query = '\n    query read1 {\n      user(id:1) { address { city } }\n    }\n    query read2 {\n      user1 { address { city } }\n      user2 { address { city } }\n      user3 { address { city } }\n    }\n    '

    def should_ignore(ignore: IgnoreContext) -> bool:
        if False:
            print('Hello World!')
        return ignore.field_args.get('id') == 1
    (errors, result) = run_query(query, 10, should_ignore=should_ignore)
    expected = {'read1': 0, 'read2': 2}
    assert not errors
    assert result == expected

def test_should_ignore_field_by_float_argument():
    if False:
        for i in range(10):
            print('nop')
    query = '\n    query read1 {\n      user(age:10.5) { address { city } }\n    }\n    query read2 {\n      user1 { address { city } }\n      user2 { address { city } }\n      user3 { address { city } }\n    }\n    '

    def should_ignore(ignore: IgnoreContext) -> bool:
        if False:
            print('Hello World!')
        return ignore.field_args.get('age') == 10.5
    (errors, result) = run_query(query, 10, should_ignore=should_ignore)
    expected = {'read1': 0, 'read2': 2}
    assert not errors
    assert result == expected

def test_should_ignore_field_by_bool_argument():
    if False:
        print('Hello World!')
    query = '\n    query read1 {\n      user(isCool:false) { address { city } }\n    }\n    query read2 {\n      user1 { address { city } }\n      user2 { address { city } }\n      user3 { address { city } }\n    }\n    '

    def should_ignore(ignore: IgnoreContext) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return ignore.field_args.get('isCool') is False
    (errors, result) = run_query(query, 10, should_ignore=should_ignore)
    expected = {'read1': 0, 'read2': 2}
    assert not errors
    assert result == expected

def test_should_ignore_field_by_name_and_str_argument():
    if False:
        i = 10
        return i + 15
    query = '\n    query read1 {\n      user(name:"matt") { address { city } }\n    }\n    query read2 {\n      user1 { address { city } }\n      user2 { address { city } }\n      user3 { address { city } }\n    }\n    '

    def should_ignore(ignore: IgnoreContext) -> bool:
        if False:
            return 10
        return ignore.field_args.get('name') == 'matt'
    (errors, result) = run_query(query, 10, should_ignore=should_ignore)
    expected = {'read1': 0, 'read2': 2}
    assert not errors
    assert result == expected

def test_should_ignore_field_by_list_argument():
    if False:
        while True:
            i = 10
    query = '\n    query read1 {\n      users(names:["matt","andy"]) { address { city } }\n    }\n    query read2 {\n      user1 { address { city } }\n      user2 { address { city } }\n      user3 { address { city } }\n    }\n    '

    def should_ignore(ignore: IgnoreContext) -> bool:
        if False:
            while True:
                i = 10
        return 'matt' in ignore.field_args.get('names', [])
    (errors, result) = run_query(query, 10, should_ignore=should_ignore)
    expected = {'read1': 0, 'read2': 2}
    assert not errors
    assert result == expected

def test_should_ignore_field_by_object_argument():
    if False:
        return 10
    query = '\n    query read1 {\n      cat(bio:{\n        name:"Momo",\n        ownerName:"Tommy"\n      }) { name }\n    }\n    query read2 {\n      user1 { address { city } }\n      user2 { address { city } }\n      user3 { address { city } }\n    }\n    '

    def should_ignore(ignore: IgnoreContext) -> bool:
        if False:
            print('Hello World!')
        return ignore.field_args.get('bio', {}).get('name') == 'Momo'
    (errors, result) = run_query(query, 10, should_ignore=should_ignore)
    expected = {'read1': 0, 'read2': 2}
    assert not errors
    assert result == expected

def test_should_work_as_extension():
    if False:
        for i in range(10):
            print('nop')
    query = '{\n    user {\n      pets {\n        owner {\n          pets {\n            owner {\n              pets {\n                name\n              }\n            }\n          }\n        }\n      }\n    }\n    }\n    '

    def should_ignore(ignore: IgnoreContext) -> bool:
        if False:
            while True:
                i = 10
        return False
    schema = strawberry.Schema(Query, extensions=[QueryDepthLimiter(max_depth=4, should_ignore=should_ignore)])
    result = schema.execute_sync(query)
    assert len(result.errors) == 1
    assert result.errors[0].message == "'anonymous' exceeds maximum operation depth of 4"