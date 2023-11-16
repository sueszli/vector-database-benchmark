from typing import List, Optional, Union
import strawberry
from strawberry.annotation import StrawberryAnnotation
from strawberry.type import StrawberryOptional
from strawberry.unset import UnsetType

def test_basic_optional():
    if False:
        return 10
    annotation = StrawberryAnnotation(Optional[str])
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryOptional)
    assert resolved.of_type is str
    assert resolved == StrawberryOptional(of_type=str)
    assert resolved == Optional[str]

def test_optional_with_unset():
    if False:
        print('Hello World!')
    annotation = StrawberryAnnotation(Union[UnsetType, Optional[str]])
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryOptional)
    assert resolved.of_type is str
    assert resolved == StrawberryOptional(of_type=str)
    assert resolved == Optional[str]

def test_optional_with_unset_as_union():
    if False:
        for i in range(10):
            print('nop')
    annotation = StrawberryAnnotation(Union[UnsetType, None, str])
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryOptional)
    assert resolved.of_type is str
    assert resolved == StrawberryOptional(of_type=str)
    assert resolved == Optional[str]

def test_optional_list():
    if False:
        for i in range(10):
            print('nop')
    annotation = StrawberryAnnotation(Optional[List[bool]])
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryOptional)
    assert resolved.of_type == List[bool]
    assert resolved == StrawberryOptional(of_type=List[bool])
    assert resolved == Optional[List[bool]]

def test_optional_optional():
    if False:
        print('Hello World!')
    'Optional[Optional[...]] is squashed by Python to just Optional[...]'
    annotation = StrawberryAnnotation(Optional[Optional[bool]])
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryOptional)
    assert resolved.of_type is bool
    assert resolved == StrawberryOptional(of_type=bool)
    assert resolved == Optional[Optional[bool]]
    assert resolved == Optional[bool]

def test_optional_union():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class CoolType:
        foo: float

    @strawberry.type
    class UncoolType:
        bar: bool
    annotation = StrawberryAnnotation(Optional[Union[CoolType, UncoolType]])
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryOptional)
    assert resolved.of_type == Union[CoolType, UncoolType]
    assert resolved == StrawberryOptional(of_type=Union[CoolType, UncoolType])
    assert resolved == Optional[Union[CoolType, UncoolType]]

def test_type_add_type_definition_with_fields():
    if False:
        print('Hello World!')

    @strawberry.type
    class Query:
        name: Optional[str]
        age: Optional[int]
    definition = Query.__strawberry_definition__
    assert definition.name == 'Query'
    [field1, field2] = definition.fields
    assert field1.python_name == 'name'
    assert field1.graphql_name is None
    assert isinstance(field1.type, StrawberryOptional)
    assert field1.type.of_type is str
    assert field2.python_name == 'age'
    assert field2.graphql_name is None
    assert isinstance(field2.type, StrawberryOptional)
    assert field2.type.of_type is int

def test_passing_custom_names_to_fields():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class Query:
        x: Optional[str] = strawberry.field(name='name')
        y: Optional[int] = strawberry.field(name='age')
    definition = Query.__strawberry_definition__
    assert definition.name == 'Query'
    [field1, field2] = definition.fields
    assert field1.python_name == 'x'
    assert field1.graphql_name == 'name'
    assert isinstance(field1.type, StrawberryOptional)
    assert field1.type.of_type is str
    assert field2.python_name == 'y'
    assert field2.graphql_name == 'age'
    assert isinstance(field2.type, StrawberryOptional)
    assert field2.type.of_type is int

def test_passing_nothing_to_fields():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Query:
        name: Optional[str] = strawberry.field()
        age: Optional[int] = strawberry.field()
    definition = Query.__strawberry_definition__
    assert definition.name == 'Query'
    [field1, field2] = definition.fields
    assert field1.python_name == 'name'
    assert field1.graphql_name is None
    assert isinstance(field1.type, StrawberryOptional)
    assert field1.type.of_type is str
    assert field2.python_name == 'age'
    assert field2.graphql_name is None
    assert isinstance(field2.type, StrawberryOptional)
    assert field2.type.of_type is int

def test_resolver_fields():
    if False:
        print('Hello World!')

    @strawberry.type
    class Query:

        @strawberry.field
        def name(self) -> Optional[str]:
            if False:
                print('Hello World!')
            return 'Name'
    definition = Query.__strawberry_definition__
    assert definition.name == 'Query'
    [field] = definition.fields
    assert field.python_name == 'name'
    assert field.graphql_name is None
    assert isinstance(field.type, StrawberryOptional)
    assert field.type.of_type is str

def test_resolver_fields_arguments():
    if False:
        i = 10
        return i + 15

    @strawberry.type
    class Query:

        @strawberry.field
        def name(self, argument: Optional[str]) -> Optional[str]:
            if False:
                i = 10
                return i + 15
            return 'Name'
    definition = Query.__strawberry_definition__
    assert definition.name == 'Query'
    [field] = definition.fields
    assert field.python_name == 'name'
    assert field.graphql_name is None
    assert isinstance(field.type, StrawberryOptional)
    assert field.type.of_type is str
    [argument] = field.arguments
    assert argument.python_name == 'argument'
    assert argument.graphql_name is None
    assert isinstance(argument.type, StrawberryOptional)
    assert argument.type.of_type is str