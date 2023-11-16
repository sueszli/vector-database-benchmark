import sys
from dataclasses import dataclass
from textwrap import dedent
from typing import Optional
import pytest
import strawberry

def test_named_union():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class A:
        a: int

    @strawberry.type
    class B:
        b: int
    with pytest.deprecated_call(match='Passing types to `strawberry.union` is deprecated'):
        Result = strawberry.union('Result', (A, B))

    @strawberry.type
    class Query:
        ab: Result = strawberry.field(default_factory=lambda : A(a=5))
    schema = strawberry.Schema(query=Query)
    query = '{\n        __type(name: "Result") {\n            kind\n            description\n        }\n\n        ab {\n            __typename,\n\n            ... on A {\n                a\n            }\n        }\n    }'
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data['ab'] == {'__typename': 'A', 'a': 5}
    assert result.data['__type'] == {'kind': 'UNION', 'description': None}

def test_named_union_description():
    if False:
        return 10

    @strawberry.type
    class A:
        a: int

    @strawberry.type
    class B:
        b: int
    with pytest.deprecated_call(match='Passing types to `strawberry.union` is deprecated'):
        Result = strawberry.union('Result', (A, B), description='Example Result')

    @strawberry.type
    class Query:
        ab: Result = strawberry.field(default_factory=lambda : A(a=5))
    schema = strawberry.Schema(query=Query)
    query = '{\n        __type(name: "Result") {\n            kind\n            description\n        }\n\n        ab {\n            __typename,\n\n            ... on A {\n                a\n            }\n        }\n    }'
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data['ab'] == {'__typename': 'A', 'a': 5}
    assert result.data['__type'] == {'kind': 'UNION', 'description': 'Example Result'}

def test_can_use_union_in_optional():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class A:
        a: int

    @strawberry.type
    class B:
        b: int
    with pytest.deprecated_call(match='Passing types to `strawberry.union` is deprecated'):
        Result = strawberry.union('Result', (A, B))

    @strawberry.type
    class Query:
        ab: Optional[Result] = None
    schema = strawberry.Schema(query=Query)
    query = '{\n        __type(name: "Result") {\n            kind\n            description\n        }\n\n        ab {\n            __typename,\n\n            ... on A {\n                a\n            }\n        }\n    }'
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data['ab'] is None

def test_union_used_multiple_times():
    if False:
        while True:
            i = 10

    @strawberry.type
    class A:
        a: int

    @strawberry.type
    class B:
        b: int
    with pytest.deprecated_call(match='Passing types to `strawberry.union` is deprecated'):
        MyUnion = strawberry.union('MyUnion', types=(A, B))

    @strawberry.type
    class Query:
        field1: MyUnion
        field2: MyUnion
    schema = strawberry.Schema(query=Query)
    assert schema.as_str() == dedent('        type A {\n          a: Int!\n        }\n\n        type B {\n          b: Int!\n        }\n\n        union MyUnion = A | B\n\n        type Query {\n          field1: MyUnion!\n          field2: MyUnion!\n        }')

def test_union_explicit_type_resolution():
    if False:
        for i in range(10):
            print('nop')

    @dataclass
    class ADataclass:
        a: int

    @strawberry.type
    class A:
        a: int

        @classmethod
        def is_type_of(cls, obj, _info) -> bool:
            if False:
                print('Hello World!')
            return isinstance(obj, ADataclass)

    @strawberry.type
    class B:
        b: int
    with pytest.deprecated_call(match='Passing types to `strawberry.union` is deprecated'):
        MyUnion = strawberry.union('MyUnion', types=(A, B))

    @strawberry.type
    class Query:

        @strawberry.field
        def my_field(self) -> MyUnion:
            if False:
                return 10
            return ADataclass(a=1)
    schema = strawberry.Schema(query=Query)
    query = '{ myField { __typename, ... on A { a }, ... on B { b } } }'
    result = schema.execute_sync(query)
    assert not result.errors
    assert result.data == {'myField': {'__typename': 'A', 'a': 1}}

@pytest.mark.skipif(sys.version_info < (3, 10), reason='pipe syntax for union is only available on python 3.10+')
def test_union_optional_with_or_operator():
    if False:
        print('Hello World!')
    '\n    Verify that the `|` operator is supported when annotating unions as\n    optional in schemas.\n    '

    @strawberry.type
    class Cat:
        name: str

    @strawberry.type
    class Dog:
        name: str
    with pytest.deprecated_call(match='Passing types to `strawberry.union` is deprecated'):
        animal_union = strawberry.union('Animal', (Cat, Dog))

    @strawberry.type
    class Query:

        @strawberry.field
        def animal(self) -> animal_union | None:
            if False:
                i = 10
                return i + 15
            return None
    schema = strawberry.Schema(query=Query)
    query = '{\n        animal {\n            __typename\n        }\n    }'
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data['animal'] is None