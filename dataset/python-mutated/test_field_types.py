from enum import Enum
from typing import List, Optional, TypeVar
import strawberry
from strawberry.annotation import StrawberryAnnotation
from strawberry.field import StrawberryField
from strawberry.union import StrawberryUnion

def test_enum():
    if False:
        print('Hello World!')

    @strawberry.enum
    class Egnum(Enum):
        a = 'A'
        b = 'B'
    annotation = StrawberryAnnotation(Egnum)
    field = StrawberryField(type_annotation=annotation)
    assert field.type is Egnum._enum_definition

def test_forward_reference():
    if False:
        i = 10
        return i + 15
    global RefForward
    annotation = StrawberryAnnotation('RefForward', namespace=globals())
    field = StrawberryField(type_annotation=annotation)

    @strawberry.type
    class RefForward:
        ref: int
    assert field.type is RefForward
    del RefForward

def test_list():
    if False:
        print('Hello World!')
    annotation = StrawberryAnnotation(List[int])
    field = StrawberryField(type_annotation=annotation)
    assert field.type == List[int]

def test_literal():
    if False:
        for i in range(10):
            print('nop')
    annotation = StrawberryAnnotation(bool)
    field = StrawberryField(type_annotation=annotation)
    assert field.type is bool

def test_object():
    if False:
        return 10

    @strawberry.type
    class TypeyType:
        value: str
    annotation = StrawberryAnnotation(TypeyType)
    field = StrawberryField(type_annotation=annotation)
    assert field.type is TypeyType

def test_optional():
    if False:
        while True:
            i = 10
    annotation = StrawberryAnnotation(Optional[float])
    field = StrawberryField(type_annotation=annotation)
    assert field.type == Optional[float]

def test_type_var():
    if False:
        print('Hello World!')
    T = TypeVar('T')
    annotation = StrawberryAnnotation(T)
    field = StrawberryField(type_annotation=annotation)
    assert field.type == T

def test_union():
    if False:
        for i in range(10):
            print('nop')

    @strawberry.type
    class Un:
        fi: int

    @strawberry.type
    class Ion:
        eld: float
    union = StrawberryUnion(name='UnionName', type_annotations=(StrawberryAnnotation(Un), StrawberryAnnotation(Ion)))
    annotation = StrawberryAnnotation(union)
    field = StrawberryField(type_annotation=annotation)
    assert field.type is union