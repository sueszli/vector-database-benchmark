import textwrap
from typing import Generic, List, NewType, TypeVar
from typing_extensions import Annotated
import pytest
import strawberry
from strawberry.enum import EnumDefinition
from strawberry.lazy_type import LazyType
from strawberry.schema.config import StrawberryConfig
from strawberry.type import StrawberryList, StrawberryOptional
from strawberry.union import StrawberryUnion
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
Enum = EnumDefinition(None, name='Enum', values=[], description=None)
CustomInt = strawberry.scalar(NewType('CustomInt', int))

@strawberry.type
class TypeA:
    name: str

@strawberry.type
class TypeB:
    age: int

@pytest.mark.parametrize(('types', 'expected_name'), [([StrawberryList(str)], 'StrListExample'), ([StrawberryList(StrawberryList(str))], 'StrListListExample'), ([StrawberryOptional(StrawberryList(str))], 'StrListOptionalExample'), ([StrawberryList(StrawberryOptional(str))], 'StrOptionalListExample'), ([StrawberryList(Enum)], 'EnumListExample'), ([StrawberryUnion('Union', (TypeA, TypeB))], 'UnionExample'), ([TypeA], 'TypeAExample'), ([CustomInt], 'CustomIntExample'), ([TypeA, TypeB], 'TypeATypeBExample'), ([TypeA, LazyType['TypeB', 'test_names']], 'TypeATypeBExample'), ([TypeA, Annotated['TypeB', strawberry.lazy('test_names')]], 'TypeATypeBExample')])
def test_name_generation(types, expected_name):
    if False:
        for i in range(10):
            print('nop')
    config = StrawberryConfig()

    @strawberry.type
    class Example(Generic[T]):
        a: T
    type_definition = Example.__strawberry_definition__
    assert config.name_converter.from_generic(type_definition, types) == expected_name

def test_nested_generics():
    if False:
        return 10
    config = StrawberryConfig()

    @strawberry.type
    class Edge(Generic[T]):
        node: T

    @strawberry.type
    class Connection(Generic[T]):
        edges: List[T]
    type_definition = Connection.__strawberry_definition__
    assert config.name_converter.from_generic(type_definition, [Edge[int]]) == 'IntEdgeConnection'

def test_nested_generics_aliases_with_schema():
    if False:
        i = 10
        return i + 15
    'This tests is similar to the previous test, but it also tests against\n    the schema, since the resolution of the type name might be different.'
    config = StrawberryConfig()

    @strawberry.type
    class Value(Generic[T]):
        value: T

    @strawberry.type
    class DictItem(Generic[K, V]):
        key: K
        value: V
    type_definition = Value.__strawberry_definition__
    assert config.name_converter.from_generic(type_definition, [StrawberryList(DictItem[int, str])]) == 'IntStrDictItemListValue'

    @strawberry.type
    class Query:
        d: Value[List[DictItem[int, str]]]
    schema = strawberry.Schema(query=Query)
    expected = textwrap.dedent('\n        type IntStrDictItem {\n          key: Int!\n          value: String!\n        }\n\n        type IntStrDictItemListValue {\n          value: [IntStrDictItem!]!\n        }\n\n        type Query {\n          d: IntStrDictItemListValue!\n        }\n        ').strip()
    assert str(schema) == expected