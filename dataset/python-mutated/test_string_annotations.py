from typing import List, Optional, TypeVar
import strawberry
from strawberry.annotation import StrawberryAnnotation
from strawberry.type import StrawberryList, StrawberryOptional, StrawberryTypeVar

def test_basic_string():
    if False:
        i = 10
        return i + 15
    annotation = StrawberryAnnotation('str')
    resolved = annotation.resolve()
    assert resolved is str

def test_list_of_string():
    if False:
        return 10
    annotation = StrawberryAnnotation(List['int'])
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryList)
    assert resolved.of_type is int
    assert resolved == StrawberryList(of_type=int)
    assert resolved == List[int]

def test_list_of_string_of_type():
    if False:
        while True:
            i = 10

    @strawberry.type
    class NameGoesHere:
        foo: bool
    annotation = StrawberryAnnotation(List['NameGoesHere'], namespace=locals())
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryList)
    assert resolved.of_type is NameGoesHere
    assert resolved == StrawberryList(of_type=NameGoesHere)
    assert resolved == List[NameGoesHere]

def test_optional_of_string():
    if False:
        i = 10
        return i + 15
    annotation = StrawberryAnnotation(Optional['bool'])
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryOptional)
    assert resolved.of_type is bool
    assert resolved == StrawberryOptional(of_type=bool)
    assert resolved == Optional[bool]

def test_string_of_object():
    if False:
        while True:
            i = 10

    @strawberry.type
    class StrType:
        thing: int
    annotation = StrawberryAnnotation('StrType', namespace=locals())
    resolved = annotation.resolve()
    assert resolved is StrType

def test_string_of_type_var():
    if False:
        i = 10
        return i + 15
    T = TypeVar('T')
    annotation = StrawberryAnnotation('T', namespace=locals())
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryTypeVar)
    assert resolved.type_var is T
    assert resolved == T

def test_string_of_list():
    if False:
        while True:
            i = 10
    namespace = {**locals(), **globals()}
    annotation = StrawberryAnnotation('List[float]', namespace=namespace)
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryList)
    assert resolved.of_type is float
    assert resolved == StrawberryList(of_type=float)
    assert resolved == List[float]

def test_string_of_list_of_type():
    if False:
        while True:
            i = 10

    @strawberry.type
    class BlahBlah:
        foo: bool
    namespace = {**locals(), **globals()}
    annotation = StrawberryAnnotation('List[BlahBlah]', namespace=namespace)
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryList)
    assert resolved.of_type is BlahBlah
    assert resolved == StrawberryList(of_type=BlahBlah)
    assert resolved == List[BlahBlah]

def test_string_of_optional():
    if False:
        for i in range(10):
            print('nop')
    namespace = {**locals(), **globals()}
    annotation = StrawberryAnnotation('Optional[int]', namespace=namespace)
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryOptional)
    assert resolved.of_type is int
    assert resolved == StrawberryOptional(of_type=int)
    assert resolved == Optional[int]

def test_basic_types():
    if False:
        while True:
            i = 10

    @strawberry.type
    class Query:
        name: 'str'
        age: 'int'
    definition = Query.__strawberry_definition__
    assert definition.name == 'Query'
    [field1, field2] = definition.fields
    assert field1.python_name == 'name'
    assert field1.type is str
    assert field2.python_name == 'age'
    assert field2.type is int

def test_optional():
    if False:
        print('Hello World!')

    @strawberry.type
    class Query:
        name: 'Optional[str]'
        age: 'Optional[int]'
    definition = Query.__strawberry_definition__
    assert definition.name == 'Query'
    [field1, field2] = definition.fields
    assert field1.python_name == 'name'
    assert isinstance(field1.type, StrawberryOptional)
    assert field1.type.of_type is str
    assert field2.python_name == 'age'
    assert isinstance(field2.type, StrawberryOptional)
    assert field2.type.of_type is int

def test_basic_list():
    if False:
        return 10

    @strawberry.type
    class Query:
        names: 'List[str]'
    definition = Query.__strawberry_definition__
    assert definition.name == 'Query'
    [field] = definition.fields
    assert field.python_name == 'names'
    assert isinstance(field.type, StrawberryList)
    assert field.type.of_type is str

def test_list_of_types():
    if False:
        return 10
    global User

    @strawberry.type
    class User:
        name: str

    @strawberry.type
    class Query:
        users: 'List[User]'
    definition = Query.__strawberry_definition__
    assert definition.name == 'Query'
    [field] = definition.fields
    assert field.python_name == 'users'
    assert isinstance(field.type, StrawberryList)
    assert field.type.of_type is User
    del User