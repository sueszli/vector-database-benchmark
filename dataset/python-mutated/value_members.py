"""
Storage format for values from data file entries.
Data from ReadMembers is supposed to be transferred
to these objects for easier handling during the conversion
process and advanced features like creating diffs.

Quick usage guide on when to use which ValueMember:
    - IntMember, FloatMember, BooleanMember and StringMember: should
      be self explanatory.
    - IDMember: References to other structures in form of identifiers.
                Also useful for flags with more than two options.
    - BitfieldMember: Value is used as a bitfield.
    - ContainerMember: For modelling specific substructures. ContainerMembers
                       can store members with different types. However, the
                       member names must be unique.
                       (e.g. a unit object)
    - ArrayMember: Stores a list of members with uniform type. Can be used
                   when repeating substructures appear in a data file.
                   (e.g. multiple unit objects, list of coordinates)
"""
from __future__ import annotations
import typing
from enum import Enum
from math import isclose
from abc import ABC, abstractmethod
from .dynamic_loader import DynamicLoader

class ValueMember(ABC):
    """
    Stores a value member from a data file.
    """
    __slots__ = ('_name', '_value')

    def __init__(self, name: str):
        if False:
            while True:
                i = 10
        self._name = name
        self._value = None

    @property
    def name(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns the name of the member.\n        '
        return self._name

    @property
    def value(self) -> typing.Any:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the value of a member.\n        '
        return self._value

    @abstractmethod
    def get_type(self) -> StorageType:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the type of a member.\n        '

    @abstractmethod
    def diff(self, other: ValueMember) -> ValueMember:
        if False:
            return 10
        "\n        Returns a new member object that contains the diff between\n        self's and other's values.\n\n        If they are equal, return a NoDiffMember.\n        "

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'{self.__class__.__name__}<{self.name}>'

class IntMember(ValueMember):
    """
    Stores numeric integer values.
    """

    def __init__(self, name: str, value: typing.Union[int, float]):
        if False:
            i = 10
            return i + 15
        super().__init__(name)
        self._value = int(value)

    def get_type(self) -> StorageType:
        if False:
            print('Hello World!')
        return StorageType.INT_MEMBER

    def diff(self, other: IntMember) -> typing.Union[NoDiffMember, IntMember]:
        if False:
            while True:
                i = 10
        if self.get_type() is other.get_type():
            if self.value == other.value:
                return NoDiffMember(self.name, self)
            else:
                diff_value = other.value - self.value
                return IntMember(self.name, diff_value)
        else:
            raise TypeError(f'type {type(self)} member cannot be diffed with type {type(other)}')

class FloatMember(ValueMember):
    """
    Stores numeric floating point values.
    """

    def __init__(self, name: str, value: typing.Union[int, float]):
        if False:
            print('Hello World!')
        super().__init__(name)
        self._value = float(value)

    def get_type(self) -> StorageType:
        if False:
            i = 10
            return i + 15
        return StorageType.FLOAT_MEMBER

    def diff(self, other: FloatMember) -> typing.Union[NoDiffMember, FloatMember]:
        if False:
            while True:
                i = 10
        if self.get_type() is other.get_type():
            if isclose(self.value, other.value, rel_tol=1e-07):
                return NoDiffMember(self.name, self)
            else:
                diff_value = other.value - self.value
                return FloatMember(self.name, diff_value)
        else:
            raise TypeError(f'type {type(self)} member cannot be diffed with type {type(other)}')

class BooleanMember(ValueMember):
    """
    Stores boolean values.
    """

    def __init__(self, name: str, value: bool):
        if False:
            return 10
        super().__init__(name)
        self._value = bool(value)

    def get_type(self) -> StorageType:
        if False:
            i = 10
            return i + 15
        return StorageType.BOOLEAN_MEMBER

    def diff(self, other: BooleanMember) -> typing.Union[NoDiffMember, BooleanMember]:
        if False:
            while True:
                i = 10
        if self.get_type() is other.get_type():
            if self.value == other.value:
                return NoDiffMember(self.name, self)
            else:
                return BooleanMember(self.name, other.value)
        else:
            raise TypeError(f'type {type(self)} member cannot be diffed with type {type(other)}')

class IDMember(ValueMember):
    """
    Stores references to media/resource IDs.
    """

    def __init__(self, name: str, value: int):
        if False:
            print('Hello World!')
        super().__init__(name)
        self._value = int(value)

    def get_type(self) -> StorageType:
        if False:
            i = 10
            return i + 15
        return StorageType.ID_MEMBER

    def diff(self, other: IDMember) -> typing.Union[NoDiffMember, IDMember]:
        if False:
            for i in range(10):
                print('nop')
        if self.get_type() is other.get_type():
            if self.value == other.value:
                return NoDiffMember(self.name, self)
            else:
                return IDMember(self.name, other.value)
        else:
            raise TypeError(f'type {type(self)} member cannot be diffed with type {type(other)}')

class BitfieldMember(ValueMember):
    """
    Stores bit field members.
    """

    def __init__(self, name: str, value: int):
        if False:
            while True:
                i = 10
        super().__init__(name)
        self._value = value

    def get_value_at_pos(self, pos: int) -> bool:
        if False:
            while True:
                i = 10
        '\n        Return the boolean value stored at a specific position\n        in the bitfield.\n\n        :param pos: Position in the bitfield, starting with the least significant bit.\n        :type pos: int\n        '
        return bool(self.value & 2 ** pos)

    def get_type(self) -> StorageType:
        if False:
            i = 10
            return i + 15
        return StorageType.BITFIELD_MEMBER

    def diff(self, other: BitfieldMember) -> typing.Union[NoDiffMember, BitfieldMember]:
        if False:
            while True:
                i = 10
        "\n        Uses XOR to determine which bits are different in 'other'.\n        "
        if self.get_type() is other.get_type():
            if self.value == other.value:
                return NoDiffMember(self.name, self)
            else:
                difference = self.value ^ other.value
                return BitfieldMember(self.name, difference)
        else:
            raise TypeError(f'type {type(self)} member cannot be diffed with type {type(other)}')

    def __len__(self):
        if False:
            return 10
        return len(self.value)

class StringMember(ValueMember):
    """
    Stores string values.
    """

    def __init__(self, name: str, value: StringMember):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name)
        self._value = str(value)

    def get_type(self) -> StorageType:
        if False:
            return 10
        return StorageType.STRING_MEMBER

    def diff(self, other: StringMember) -> typing.Union[NoDiffMember, StringMember]:
        if False:
            for i in range(10):
                print('nop')
        if self.get_type() is other.get_type():
            if self.value == other.value:
                return NoDiffMember(self.name, self)
            else:
                return StringMember(self.name, other.value)
        else:
            raise TypeError(f'type {type(self)} member cannot be diffed with type {type(other)}')

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.value)

class ContainerMember(ValueMember):
    """
    Stores multiple members as key-value pairs.

    The name of the members are the keys, the member objects
    are the value of the dict.
    """

    def __init__(self, name: str, submembers: list[typing.Union[IntMember, FloatMember, BooleanMember, IDMember, BitfieldMember, StringMember, ArrayMember, ContainerMember]]):
        if False:
            print('Hello World!')
        '\n        :param submembers: Stored members as a list or dict\n        :type submembers: list, dict\n        '
        super().__init__(name)
        self._value = {}
        if isinstance(submembers, (dict, DynamicLoader)):
            self._value = submembers
        else:
            self._create_dict(submembers)

    def get_type(self) -> StorageType:
        if False:
            print('Hello World!')
        return StorageType.CONTAINER_MEMBER

    def diff(self, other: ContainerMember) -> typing.Union[NoDiffMember, ContainerMember]:
        if False:
            while True:
                i = 10
        if self.get_type() is other.get_type():
            diff_dict = {}
            other_dict = other.value
            for key in self.value.keys():
                if key in other.value.keys():
                    diff_value = self.value[key].diff(other_dict[key])
                else:
                    diff_value = RightMissingMember(key, self.value[key])
                diff_dict.update({key: diff_value})
            for key in other.value.keys():
                if key not in self.value.keys():
                    diff_value = LeftMissingMember(key, other_dict[key])
                    diff_dict.update({key: diff_value})
            if all((isinstance(member, NoDiffMember) for member in diff_dict.values())):
                return NoDiffMember(self.name, self)
            return ContainerMember(self.name, diff_dict)
        else:
            raise TypeError(f'type {type(self)} member cannot be diffed with type {type(other)}')

    def _create_dict(self, member_list: list[typing.Union[IntMember, FloatMember, BooleanMember, IDMember, BitfieldMember, StringMember, ArrayMember, ContainerMember]]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Creates the dict from the member list passed to __init__.\n        '
        for member in member_list:
            self._value.update({member.name: member})

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        '\n        Short command for getting a member in the container.\n        '
        return self.value[key]

    def __len__(self):
        if False:
            return 10
        return len(self.value)

class ArrayMember(ValueMember):
    """
    Stores an ordered list of members with the same type.
    """
    __slots__ = '_allowed_member_type'

    def __init__(self, name: str, allowed_member_type: StorageType, members: list[typing.Union[IntMember, FloatMember, BooleanMember, IDMember, BitfieldMember, StringMember, ArrayMember, ContainerMember]]):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name)
        self._value = members
        self._allowed_member_type = allowed_member_type
        for member in members:
            if not isinstance(member, (NoDiffMember, LeftMissingMember, RightMissingMember)):
                if member.get_type() is not self._allowed_member_type:
                    raise TypeError('%s has type %s, but this ArrayMember only allows %s' % (member, member.get_type(), allowed_member_type))

    def get_type(self) -> StorageType:
        if False:
            while True:
                i = 10
        if self._allowed_member_type is StorageType.INT_MEMBER:
            return StorageType.ARRAY_INT
        elif self._allowed_member_type is StorageType.FLOAT_MEMBER:
            return StorageType.ARRAY_FLOAT
        elif self._allowed_member_type is StorageType.BOOLEAN_MEMBER:
            return StorageType.ARRAY_BOOL
        elif self._allowed_member_type is StorageType.ID_MEMBER:
            return StorageType.ARRAY_ID
        elif self._allowed_member_type is StorageType.BITFIELD_MEMBER:
            return StorageType.ARRAY_BITFIELD
        elif self._allowed_member_type is StorageType.STRING_MEMBER:
            return StorageType.ARRAY_STRING
        elif self._allowed_member_type is StorageType.CONTAINER_MEMBER:
            return StorageType.ARRAY_CONTAINER
        raise TypeError(f'{self} has no valid member type')

    def get_container(self, key_member_name: str, force_not_found: bool=False, force_duplicate: bool=False) -> ContainerMember:
        if False:
            print('Hello World!')
        '\n        Returns a ContainerMember generated from an array with type ARRAY_CONTAINER.\n        It uses the values of the members with the specified name as keys.\n        By default, this method raises an exception if a member with this\n        name does not exist or the same key is used twice.\n\n        :param key_member_name: A member in the containers whos value is used as the key.\n        :type key_member_name: str\n        :param force_not_found: Do not raise an exception if the member is not found.\n        :type force_not_found: bool\n        :param force_duplicate: Do not raise an exception if the same key value is used twice.\n        :type force_duplicate: bool\n        '
        if self.get_type() is not StorageType.ARRAY_CONTAINER:
            raise TypeError("%s: Container can only be generated from arrays with type 'contarray', not %s" % (self, self.get_type()))
        member_dict = {}
        for container in self.value:
            if key_member_name not in container.value.keys():
                if force_not_found:
                    continue
                raise KeyError('%s: Container %s has no member called %s' % (self, container, key_member_name))
            key_member_value = container[key_member_name].value
            if key_member_value in member_dict.keys():
                if force_duplicate:
                    continue
                raise KeyError('%s: Duplicate key %s for container member %s' % (self, key_member_value, key_member_name))
            member_dict.update({key_member_value: container})
        return ContainerMember(self.name, member_dict)

    def diff(self, other: ArrayMember) -> typing.Union[NoDiffMember, ArrayMember]:
        if False:
            return 10
        if self.get_type() == other.get_type():
            diff_list = []
            other_list = other.value
            index = 0
            if len(self) <= len(other):
                while index < len(self):
                    diff_value = self.value[index].diff(other_list[index])
                    diff_list.append(diff_value)
                    index += 1
                while index < len(other):
                    diff_value = other_list[index]
                    diff_list.append(LeftMissingMember(diff_value.name, diff_value))
                    index += 1
            else:
                while index < len(other):
                    diff_value = self.value[index].diff(other_list[index])
                    diff_list.append(diff_value)
                    index += 1
                while index < len(self):
                    diff_value = self.value[index]
                    diff_list.append(RightMissingMember(diff_value.name, diff_value))
                    index += 1
            if all((isinstance(member, NoDiffMember) for member in diff_list)):
                return NoDiffMember(self.name, self)
            return ArrayMember(self.name, self._allowed_member_type, diff_list)
        else:
            raise TypeError(f'type {type(self)} member cannot be diffed with type {type(other)}')

    def __getitem__(self, key):
        if False:
            return 10
        '\n        Short command for getting a member in the array.\n        '
        return self.value[key]

    def __len__(self):
        if False:
            return 10
        return len(self.value)

class NoDiffMember(ValueMember):
    """
    Is returned when no difference between two members is found.
    """

    def __init__(self, name: str, value: ValueMember):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param value: Reference to the one of the diffed members.\n        :type value: ValueMember\n        '
        super().__init__(name)
        self._value = value

    @property
    def ref(self) -> ValueMember:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the reference to the diffed object.\n        '
        return self._value

    @property
    def value(self) -> typing.Any:
        if False:
            i = 10
            return i + 15
        '\n        Returns the value of a member.\n        '
        raise NotImplementedError(f"{type(self)} cannot have values; use 'ref' instead")

    def get_type(self) -> typing.NoReturn:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the type of a member.\n        '
        raise NotImplementedError(f'{type(self)} cannot have a type')

    def diff(self, other: typing.Any) -> typing.NoReturn:
        if False:
            while True:
                i = 10
        "\n        Returns a new member object that contains the diff between\n        self's and other's values.\n\n        If they are equal, return a NoDiffMember.\n        "
        raise NotImplementedError(f'{type(self)} cannot be diffed')

class LeftMissingMember(ValueMember):
    """
    Is returned when an array or container on the left side of
    the comparison has no member to compare. It stores the right
    side member as value.
    """

    def __init__(self, name: str, value: ValueMember):
        if False:
            for i in range(10):
                print('nop')
        "\n        :param value: Reference to the right member's object.\n        :type value: ValueMember\n        "
        super().__init__(name)
        self._value = value

    @property
    def ref(self) -> ValueMember:
        if False:
            return 10
        '\n        Returns the reference to the diffed object.\n        '
        return self._value

    @property
    def value(self) -> typing.Any:
        if False:
            print('Hello World!')
        '\n        Returns the value of a member.\n        '
        raise NotImplementedError(f"{type(self)} cannot have values; use 'ref' instead")

    def get_type(self) -> typing.NoReturn:
        if False:
            return 10
        '\n        Returns the type of a member.\n        '
        raise NotImplementedError(f'{type(self)} cannot have a type')

    def diff(self, other: typing.Any) -> typing.NoReturn:
        if False:
            i = 10
            return i + 15
        "\n        Returns a new member object that contains the diff between\n        self's and other's values.\n\n        If they are equal, return a NoDiffMember.\n        "
        raise NotImplementedError(f'{type(self)} cannot be diffed')

class RightMissingMember(ValueMember):
    """
    Is returned when an array or container on the right side of
    the comparison has no member to compare. It stores the left
    side member as value.
    """

    def __init__(self, name: str, value: ValueMember):
        if False:
            return 10
        "\n        :param value: Reference to the left member's object.\n        :type value: ValueMember\n        "
        super().__init__(name)
        self._value = value

    @property
    def ref(self) -> ValueMember:
        if False:
            print('Hello World!')
        '\n        Returns the reference to the diffed object.\n        '
        return self._value

    @property
    def value(self) -> typing.Any:
        if False:
            i = 10
            return i + 15
        '\n        Returns the value of a member.\n        '
        raise NotImplementedError(f"{type(self)} cannot have values; use 'ref' instead")

    def get_type(self) -> typing.NoReturn:
        if False:
            while True:
                i = 10
        '\n        Returns the type of a member.\n        '
        raise NotImplementedError(f'{type(self)} cannot have a type')

    def diff(self, other: typing.Any) -> typing.NoReturn:
        if False:
            print('Hello World!')
        "\n        Returns a new member object that contains the diff between\n        self's and other's values.\n\n        If they are equal, return a NoDiffMember.\n        "
        raise NotImplementedError(f'{type(self)} cannot be diffed')

class StorageType(Enum):
    """
    Types for values members.
    """
    INT_MEMBER = 'int'
    FLOAT_MEMBER = 'float'
    BOOLEAN_MEMBER = 'boolean'
    ID_MEMBER = 'id'
    BITFIELD_MEMBER = 'bitfield'
    STRING_MEMBER = 'string'
    CONTAINER_MEMBER = 'container'
    ARRAY_INT = 'intarray'
    ARRAY_FLOAT = 'floatarray'
    ARRAY_BOOL = 'boolarray'
    ARRAY_ID = 'idarray'
    ARRAY_BITFIELD = 'bitfieldarray'
    ARRAY_STRING = 'stringarray'
    ARRAY_CONTAINER = 'contarray'