import textwrap
from enum import Enum
from typing import Generic, List, Optional, TypeVar, Union
import strawberry
from strawberry.arguments import StrawberryArgument
from strawberry.custom_scalar import ScalarDefinition
from strawberry.directive import StrawberryDirective
from strawberry.enum import EnumDefinition, EnumValue
from strawberry.field import StrawberryField
from strawberry.schema.config import StrawberryConfig
from strawberry.schema.name_converter import NameConverter
from strawberry.schema_directive import Location, StrawberrySchemaDirective
from strawberry.type import StrawberryType
from strawberry.types.types import StrawberryObjectDefinition
from strawberry.union import StrawberryUnion

class AppendsNameConverter(NameConverter):

    def __init__(self, suffix: str):
        if False:
            return 10
        self.suffix = suffix
        super().__init__(auto_camel_case=True)

    def from_argument(self, argument: StrawberryArgument) -> str:
        if False:
            i = 10
            return i + 15
        return super().from_argument(argument) + self.suffix

    def from_scalar(self, scalar: ScalarDefinition) -> str:
        if False:
            i = 10
            return i + 15
        return super().from_scalar(scalar) + self.suffix

    def from_field(self, field: StrawberryField) -> str:
        if False:
            print('Hello World!')
        return super().from_field(field) + self.suffix

    def from_union(self, union: StrawberryUnion) -> str:
        if False:
            return 10
        return super().from_union(union) + self.suffix

    def from_generic(self, generic_type: StrawberryObjectDefinition, types: List[Union[StrawberryType, type]]) -> str:
        if False:
            for i in range(10):
                print('nop')
        return super().from_generic(generic_type, types) + self.suffix

    def from_interface(self, interface: StrawberryObjectDefinition) -> str:
        if False:
            print('Hello World!')
        return super().from_interface(interface) + self.suffix

    def from_directive(self, directive: Union[StrawberryDirective, StrawberrySchemaDirective]) -> str:
        if False:
            i = 10
            return i + 15
        return super().from_directive(directive) + self.suffix

    def from_input_object(self, input_type: StrawberryObjectDefinition) -> str:
        if False:
            for i in range(10):
                print('nop')
        return super().from_object(input_type) + self.suffix

    def from_object(self, object_type: StrawberryObjectDefinition) -> str:
        if False:
            while True:
                i = 10
        return super().from_object(object_type) + self.suffix

    def from_enum(self, enum: EnumDefinition) -> str:
        if False:
            while True:
                i = 10
        return super().from_enum(enum) + self.suffix

    def from_enum_value(self, enum: EnumDefinition, enum_value: EnumValue) -> str:
        if False:
            return 10
        return super().from_enum_value(enum, enum_value) + self.suffix
T = TypeVar('T')
MyScalar = strawberry.scalar(str, name='SensitiveConfiguration')

@strawberry.enum
class MyEnum(Enum):
    A = 'a'
    B = 'b'

@strawberry.type
class User:
    name: str

@strawberry.type
class Error:
    message: str

@strawberry.input
class UserInput:
    name: str

@strawberry.schema_directive(locations=[Location.FIELD_DEFINITION])
class MyDirective:
    name: str

@strawberry.interface
class Node:
    id: strawberry.ID

@strawberry.type
class MyGeneric(Generic[T]):
    value: T

@strawberry.type
class Query:

    @strawberry.field(directives=[MyDirective(name='my-directive')])
    def user(self, input: UserInput) -> Union[User, Error]:
        if False:
            i = 10
            return i + 15
        return User(name='Patrick')
    enum: MyEnum = MyEnum.A
    field: Optional[MyGeneric[str]] = None

    @strawberry.field
    def print(self, enum: MyEnum) -> str:
        if False:
            while True:
                i = 10
        return enum.value
schema = strawberry.Schema(query=Query, types=[MyScalar, Node], config=StrawberryConfig(name_converter=AppendsNameConverter('X')))

def test_name_converter():
    if False:
        i = 10
        return i + 15
    expected_schema = '\n    directive @myDirectiveX(name: String!) on FIELD_DEFINITION\n\n    type ErrorX {\n      messageX: String!\n    }\n\n    enum MyEnumX {\n      AX\n      BX\n    }\n\n    interface NodeXX {\n      idX: ID!\n    }\n\n    type QueryX {\n      enumX: MyEnumX!\n      fieldX: StrMyGenericXX\n      userX(inputX: UserInputX!): UserXErrorXX! @myDirectiveX(name: "my-directive")\n      printX(enumX: MyEnumX!): String!\n    }\n\n    scalar SensitiveConfiguration\n\n    type StrMyGenericXX {\n      valueX: String!\n    }\n\n    input UserInputX {\n      nameX: String!\n    }\n\n    type UserX {\n      nameX: String!\n    }\n\n    union UserXErrorXX = UserX | ErrorX\n    '
    assert textwrap.dedent(expected_schema).strip() == str(schema)

def test_returns_enum_with_correct_value():
    if False:
        i = 10
        return i + 15
    query = ' { enumX } '
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data == {'enumX': 'AX'}

def test_can_use_enum_value():
    if False:
        for i in range(10):
            print('nop')
    query = ' { printX(enumX: AX) } '
    result = schema.execute_sync(query, root_value=Query())
    assert not result.errors
    assert result.data == {'printX': 'a'}

def test_can_use_enum_value_with_variable():
    if False:
        for i in range(10):
            print('nop')
    query = ' query ($enum: MyEnumX!) { printX(enumX: $enum) } '
    result = schema.execute_sync(query, root_value=Query(), variable_values={'enum': 'AX'})
    assert not result.errors
    assert result.data == {'printX': 'a'}