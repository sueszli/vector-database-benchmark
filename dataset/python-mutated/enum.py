import dataclasses
from enum import EnumMeta
from typing import Any, Callable, Iterable, List, Mapping, Optional, TypeVar, Union, overload
from strawberry.type import StrawberryType
from .exceptions import ObjectIsNotAnEnumError

@dataclasses.dataclass
class EnumValue:
    name: str
    value: Any
    deprecation_reason: Optional[str] = None
    directives: Iterable[object] = ()
    description: Optional[str] = None

@dataclasses.dataclass
class EnumDefinition(StrawberryType):
    wrapped_cls: EnumMeta
    name: str
    values: List[EnumValue]
    description: Optional[str]
    directives: Iterable[object] = ()

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash(self.name)

    def copy_with(self, type_var_map: Mapping[str, Union[StrawberryType, type]]) -> Union[StrawberryType, type]:
        if False:
            for i in range(10):
                print('nop')
        return self

    @property
    def is_graphql_generic(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False

@dataclasses.dataclass
class EnumValueDefinition:
    value: Any
    deprecation_reason: Optional[str] = None
    directives: Iterable[object] = ()
    description: Optional[str] = None

    def __int__(self) -> int:
        if False:
            while True:
                i = 10
        return self.value

def enum_value(value: Any, deprecation_reason: Optional[str]=None, directives: Iterable[object]=(), description: Optional[str]=None) -> EnumValueDefinition:
    if False:
        print('Hello World!')
    return EnumValueDefinition(value=value, deprecation_reason=deprecation_reason, directives=directives, description=description)
EnumType = TypeVar('EnumType', bound=EnumMeta)

def _process_enum(cls: EnumType, name: Optional[str]=None, description: Optional[str]=None, directives: Iterable[object]=()) -> EnumType:
    if False:
        while True:
            i = 10
    if not isinstance(cls, EnumMeta):
        raise ObjectIsNotAnEnumError(cls)
    if not name:
        name = cls.__name__
    values = []
    for item in cls:
        item_value = item.value
        item_name = item.name
        deprecation_reason = None
        item_directives: Iterable[object] = ()
        enum_value_description = None
        if isinstance(item_value, EnumValueDefinition):
            item_directives = item_value.directives
            enum_value_description = item_value.description
            deprecation_reason = item_value.deprecation_reason
            item_value = item_value.value
            cls._value2member_map_[item_value] = item
            cls._member_map_[item_name]._value_ = item_value
        value = EnumValue(item_name, item_value, deprecation_reason=deprecation_reason, directives=item_directives, description=enum_value_description)
        values.append(value)
    cls._enum_definition = EnumDefinition(wrapped_cls=cls, name=name, values=values, description=description, directives=directives)
    return cls

@overload
def enum(_cls: EnumType, *, name: Optional[str]=None, description: Optional[str]=None, directives: Iterable[object]=()) -> EnumType:
    if False:
        i = 10
        return i + 15
    ...

@overload
def enum(_cls: None=None, *, name: Optional[str]=None, description: Optional[str]=None, directives: Iterable[object]=()) -> Callable[[EnumType], EnumType]:
    if False:
        return 10
    ...

def enum(_cls: Optional[EnumType]=None, *, name: Optional[str]=None, description: Optional[str]=None, directives: Iterable[object]=()) -> Union[EnumType, Callable[[EnumType], EnumType]]:
    if False:
        while True:
            i = 10
    'Registers the enum in the GraphQL type system.\n\n    If name is passed, the name of the GraphQL type will be\n    the value passed of name instead of the Enum class name.\n    '

    def wrap(cls: EnumType) -> EnumType:
        if False:
            return 10
        return _process_enum(cls, name, description, directives=directives)
    if not _cls:
        return wrap
    return wrap(_cls)