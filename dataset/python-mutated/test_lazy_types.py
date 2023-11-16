import enum
from typing import Generic, TypeVar
from typing_extensions import Annotated
import strawberry
from strawberry.annotation import StrawberryAnnotation
from strawberry.field import StrawberryField
from strawberry.lazy_type import LazyType
from strawberry.type import get_object_definition
from strawberry.types.fields.resolver import StrawberryResolver
from strawberry.union import StrawberryUnion, union

@strawberry.type
class LaziestType:
    something: bool

@strawberry.enum
class LazyEnum(enum.Enum):
    BREAD = 'BREAD'

def test_lazy_type():
    if False:
        return 10
    LazierType = LazyType('LaziestType', 'test_lazy_types')
    annotation = StrawberryAnnotation(LazierType)
    resolved = annotation.resolve()
    assert isinstance(resolved, LazyType)
    assert resolved is LazierType
    assert resolved.resolve_type() is LaziestType

def test_lazy_type_function():
    if False:
        print('Hello World!')
    LethargicType = Annotated['LaziestType', strawberry.lazy('test_lazy_types')]
    annotation = StrawberryAnnotation(LethargicType)
    resolved = annotation.resolve()
    assert isinstance(resolved, LazyType)
    assert resolved.resolve_type() is LaziestType

def test_lazy_type_enum():
    if False:
        i = 10
        return i + 15
    LazierType = LazyType('LazyEnum', 'test_lazy_types')
    annotation = StrawberryAnnotation(LazierType)
    resolved = annotation.resolve()
    assert isinstance(resolved, LazyType)
    assert resolved is LazierType
    assert resolved.resolve_type() is LazyEnum

def test_lazy_type_argument():
    if False:
        for i in range(10):
            print('nop')
    LazierType = LazyType('LaziestType', 'test_lazy_types')

    @strawberry.mutation
    def slack_off(emotion: LazierType) -> bool:
        if False:
            for i in range(10):
                print('nop')
        _ = emotion
        return True
    argument = slack_off.arguments[0]
    assert isinstance(argument.type, LazyType)
    assert argument.type is LazierType
    assert argument.type.resolve_type() is LaziestType

def test_lazy_type_field():
    if False:
        i = 10
        return i + 15
    LazierType = LazyType('LaziestType', 'test_lazy_types')
    annotation = StrawberryAnnotation(LazierType)
    field = StrawberryField(type_annotation=annotation)
    assert isinstance(field.type, LazyType)
    assert field.type is LazierType
    assert field.type.resolve_type() is LaziestType

def test_lazy_type_generic():
    if False:
        while True:
            i = 10
    T = TypeVar('T')

    @strawberry.type
    class GenericType(Generic[T]):
        item: T
    LazierType = LazyType('LaziestType', 'test_lazy_types')
    ResolvedType = GenericType[LazierType]
    annotation = StrawberryAnnotation(ResolvedType)
    resolved = annotation.resolve()
    definition = get_object_definition(resolved)
    assert definition
    items_field: StrawberryField = definition.fields[0]
    assert items_field.type is LazierType
    assert items_field.type.resolve_type() is LaziestType

def test_lazy_type_object():
    if False:
        return 10
    LazierType = LazyType('LaziestType', 'test_lazy_types')

    @strawberry.type
    class WaterParkFeature:
        river: LazierType
    field: StrawberryField = WaterParkFeature.__strawberry_definition__.fields[0]
    assert isinstance(field.type, LazyType)
    assert field.type is LazierType
    assert field.type.resolve_type() is LaziestType

def test_lazy_type_resolver():
    if False:
        print('Hello World!')
    LazierType = LazyType('LaziestType', 'test_lazy_types')

    def slaking_pokemon() -> LazierType:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError
    resolver = StrawberryResolver(slaking_pokemon)
    assert isinstance(resolver.type, LazyType)
    assert resolver.type is LazierType
    assert resolver.type.resolve_type() is LaziestType

def test_lazy_type_in_union():
    if False:
        print('Hello World!')
    ActiveType = LazyType('LaziestType', 'test_lazy_types')
    ActiveEnum = LazyType('LazyEnum', 'test_lazy_types')
    something = union(name='CoolUnion', types=(ActiveType, ActiveEnum))
    annotation = StrawberryAnnotation(something)
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryUnion)
    [type1, type2] = resolved.types
    assert type1 is ActiveType
    assert type2 is ActiveEnum
    assert type1.resolve_type() is LaziestType
    assert type2.resolve_type() is LazyEnum

def test_lazy_function_in_union():
    if False:
        while True:
            i = 10
    ActiveType = Annotated['LaziestType', strawberry.lazy('test_lazy_types')]
    ActiveEnum = Annotated['LazyEnum', strawberry.lazy('test_lazy_types')]
    something = union(name='CoolUnion', types=(ActiveType, ActiveEnum))
    annotation = StrawberryAnnotation(something)
    resolved = annotation.resolve()
    assert isinstance(resolved, StrawberryUnion)
    [type1, type2] = resolved.types
    assert type1.resolve_type() is LaziestType
    assert type2.resolve_type() is LazyEnum