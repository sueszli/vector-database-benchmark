"""
Nyan structs.

Simple implementation to store nyan objects and
members for usage in the converter. This is not
a real nyan^TM implementation, but rather a "dumb"
storage format.

Python does not enforce static types, so be careful
 and only use the provided functions, please. :)
"""
from __future__ import annotations
import typing
from enum import Enum
import re
from ..util.ordered_set import OrderedSet
if typing.TYPE_CHECKING:
    from openage.nyan.import_tree import ImportTree
INDENT = '    '
MAX_LINE_WIDTH = 130

class NyanObject:
    """
    Superclass for nyan objects.
    """
    __slots__ = ('name', '_fqon', '_parents', '_inherited_members', '_members', '_nested_objects', '_children')

    def __init__(self, name: str, parents: OrderedSet[NyanObject]=None, members: OrderedSet[NyanMember]=None, nested_objects: OrderedSet[NyanObject]=None):
        if False:
            print('Hello World!')
        '\n        Initializes the object and does some correctness\n        checks, for your convenience.\n        '
        self.name = name
        self._fqon: tuple[str] = (self.name,)
        self._parents: OrderedSet[NyanObject] = OrderedSet()
        self._inherited_members: OrderedSet[InheritedNyanMember] = OrderedSet()
        if parents:
            self._parents.update(parents)
        self._members: OrderedSet[NyanMember] = OrderedSet()
        if members:
            self._members.update(members)
        self._nested_objects: OrderedSet[NyanObject] = OrderedSet()
        if nested_objects:
            self._nested_objects.update(nested_objects)
            for nested_object in self._nested_objects:
                nested_object.set_fqon(f'{self._fqon}.{nested_object.get_name()}')
        self._children: OrderedSet[NyanObject] = OrderedSet()
        self._sanity_check()
        if len(self._parents) > 0:
            self._process_inheritance()

    def add_nested_object(self, new_nested_object: NyanObject) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a nested object to the nyan object.\n        '
        if not isinstance(new_nested_object, NyanObject):
            raise TypeError('nested object must have <NyanObject> type')
        if new_nested_object is self:
            raise ValueError('nyan object must not contain itself as nested object')
        self._nested_objects.add(new_nested_object)
        new_nested_object.set_fqon((*self._fqon, new_nested_object.get_name()))

    def add_member(self, new_member: NyanMember) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a member to the nyan object.\n        '
        if new_member.is_inherited():
            raise TypeError('added member cannot be inherited')
        if not isinstance(new_member, NyanMember):
            raise TypeError('added member must have <NyanMember> type')
        self._members.add(new_member)
        for child in self._children:
            inherited_member = InheritedNyanMember(new_member.get_name(), new_member.get_member_type(), self, self, None, None, 0)
            child.update_inheritance(inherited_member)

    def add_child(self, new_child: NyanObject) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Registers another object as a child.\n        '
        if not isinstance(new_child, NyanObject):
            raise TypeError('children must have <NyanObject> type')
        self._children.add(new_child)
        for member in self._members:
            inherited_member = InheritedNyanMember(member.get_name(), member.get_member_type(), self, self, None, None, 0)
            new_child.update_inheritance(inherited_member)
        for inherited in self._inherited_members:
            inherited_member = InheritedNyanMember(inherited.get_name(), inherited.get_member_type(), self, inherited.get_origin(), None, None, 0)
            new_child.update_inheritance(inherited_member)

    def has_member(self, member_name: str, origin: NyanObject=None) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns True if the NyanMember with the specified name exists.\n        '
        if origin and origin is not self:
            for inherited_member in self._inherited_members:
                if origin == inherited_member.get_origin():
                    if inherited_member.get_name() == member_name:
                        return True
        else:
            for member in self._members:
                if member.get_name() == member_name:
                    return True
        return False

    def get_fqon(self) -> tuple[str]:
        if False:
            i = 10
            return i + 15
        '\n        Returns the fqon of the nyan object.\n        '
        return self._fqon

    def get_members(self) -> OrderedSet[NyanMember]:
        if False:
            return 10
        '\n        Returns all NyanMembers of the object, including inherited members.\n        '
        return self._members.union(self._inherited_members)

    def get_member_by_name(self, member_name: str, origin: NyanObject=None) -> NyanMember:
        if False:
            while True:
                i = 10
        '\n        Returns the NyanMember with the specified name.\n        '
        if origin and origin is not self:
            for inherited_member in self._inherited_members:
                if origin == inherited_member.get_origin():
                    if inherited_member.get_name() == member_name:
                        return inherited_member
            raise ValueError(f"{repr(self)} has no member '{member_name}' with origin '{origin}'")
        for member in self._members:
            if member.get_name() == member_name:
                return member
        raise ValueError(f"{self} has no member '{member_name}'")

    def get_uninitialized_members(self) -> list:
        if False:
            while True:
                i = 10
        '\n        Returns all uninitialized NyanMembers of the object.\n        '
        uninit_members = []
        for member in self.get_members():
            if not member.is_initialized():
                uninit_members.append(member)
        return uninit_members

    def get_name(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns the name of the object.\n        '
        return self.name

    def get_nested_objects(self) -> OrderedSet[NyanObject]:
        if False:
            return 10
        '\n        Returns all nested NyanObjects of this object.\n        '
        return self._nested_objects

    def get_parents(self) -> OrderedSet[NyanObject]:
        if False:
            print('Hello World!')
        '\n        Returns all nested parents of this object.\n        '
        return self._parents

    def has_ancestor(self, nyan_object: NyanObject) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns True if the given nyan object is an ancestor\n        of this nyan object.\n        '
        for parent in self._parents:
            if parent is nyan_object:
                return True
        for parent in self._parents:
            if parent.has_ancestor(nyan_object):
                return True
        return False

    def is_abstract(self) -> bool:
        if False:
            return 10
        '\n        Returns True if any unique or inherited members are uninitialized.\n        '
        return len(self.get_uninitialized_members()) > 0

    @staticmethod
    def is_patch() -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns True if the object is a NyanPatch.\n        '
        return False

    def set_fqon(self, new_fqon: tuple[str]):
        if False:
            print('Hello World!')
        '\n        Set a new value for the fqon.\n        '
        if isinstance(new_fqon, str):
            self._fqon = new_fqon.split('.')
        elif isinstance(new_fqon, tuple):
            self._fqon = new_fqon
        else:
            raise TypeError(f'{self}: Fqon must be a tuple(str) not {type(new_fqon)}')
        for nested_object in self._nested_objects:
            nested_fqon = (*new_fqon, nested_object.get_name())
            nested_object.set_fqon(nested_fqon)

    def update_inheritance(self, new_inherited_member: InheritedNyanMember) -> None:
        if False:
            print('Hello World!')
        '\n        Add an inherited member to the object. Should only be used by\n        parent objects.\n        '
        if not self.has_ancestor(new_inherited_member.get_origin()):
            raise ValueError(f'{repr(self)}: cannot add inherited member {new_inherited_member} because {new_inherited_member.get_origin()} is not an ancestor of {repr(self)}')
        if not isinstance(new_inherited_member, InheritedNyanMember):
            raise TypeError('added member must have <InheritedNyanMember> type')
        if not self.has_member(new_inherited_member.get_name(), new_inherited_member.get_origin()):
            self._inherited_members.add(new_inherited_member)
        for child in self._children:
            inherited_member = InheritedNyanMember(new_inherited_member.get_name(), new_inherited_member.get_member_type(), self, new_inherited_member.get_origin(), None, None, 0)
            child.update_inheritance(inherited_member)

    def dump(self, indent_depth: int=0, import_tree: ImportTree=None) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns the string representation of the object.\n        '
        output_str = f'{self.get_name()}'
        output_str += self._prepare_inheritance_content(import_tree=import_tree)
        output_str += self._prepare_object_content(indent_depth, import_tree=import_tree)
        return output_str

    def _prepare_object_content(self, indent_depth: int, import_tree: ImportTree=None) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Returns a string containing the nyan object's content\n        (members, nested objects).\n\n        Subroutine of dump().\n        "
        output_str = ''
        empty = True
        if len(self._inherited_members) > 0:
            for inherited_member in self._inherited_members:
                if inherited_member.has_value():
                    empty = False
                    member_str = inherited_member.dump(indent_depth + 1, import_tree=import_tree, namespace=self.get_fqon())
                    output_str += f'{(indent_depth + 1) * INDENT}{member_str}\n'
            if not empty:
                output_str += '\n'
        if len(self._members) > 0:
            empty = False
            for member in self._members:
                if self.is_patch():
                    member_str = member.dump_short(indent_depth + 1, import_tree=import_tree, namespace=self.get_fqon())
                else:
                    member_str = member.dump(indent_depth + 1, import_tree=import_tree, namespace=self.get_fqon())
                output_str += f'{(indent_depth + 1) * INDENT}{member_str}\n'
            output_str += '\n'
        if len(self._nested_objects) > 0:
            empty = False
            for nested_object in self._nested_objects:
                nested_str = nested_object.dump(indent_depth + 1, import_tree=import_tree)
                output_str += f'{(indent_depth + 1) * INDENT}{nested_str}\n'
            output_str = output_str[:-1]
        if empty:
            output_str += f'{(indent_depth + 1) * INDENT}pass\n\n'
        return output_str

    def _prepare_inheritance_content(self, import_tree: ImportTree=None) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Returns a string containing the nyan object's inheritance set\n        in the header.\n\n        Subroutine of dump().\n        "
        output_str = '('
        if len(self._parents) > 0:
            for parent in self._parents:
                if import_tree:
                    sfqon = '.'.join(import_tree.get_alias_fqon(parent.get_fqon(), namespace=self.get_fqon()))
                else:
                    sfqon = '.'.join(parent.get_fqon())
                output_str += f'{sfqon}, '
            output_str = output_str[:-2]
        output_str += '):\n'
        return output_str

    def _process_inheritance(self) -> None:
        if False:
            print('Hello World!')
        '\n        Notify parents of the object.\n        '
        for parent in self._parents:
            parent.add_child(self)

    def _sanity_check(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Check if the object conforms to nyan grammar rules. Also does\n        a bunch of type checks.\n        '
        if not isinstance(self.name, str):
            raise TypeError(f"{repr(self)}: 'name' must be a string")
        if not re.fullmatch('[a-zA-Z_][a-zA-Z0-9_]*', self.name):
            raise SyntaxError(f"{repr(self)}: 'name' is not well-formed")
        for parent in self._parents:
            if not isinstance(parent, NyanObject):
                raise TypeError(f'{repr(self)}: {repr(parent)} must have NyanObject type')
        for member in self._members:
            if not isinstance(member, NyanMember):
                raise TypeError(f'{repr(self)}: {repr(member)} must have NyanMember type')
            if isinstance(member, InheritedNyanMember):
                raise TypeError(f'{repr(self)}: regular member {repr(member)} must not have InheritedNyanMember type')
        for nested_object in self._nested_objects:
            if not isinstance(nested_object, NyanObject):
                raise TypeError(f'{repr(self)}: {repr(nested_object)} must have NyanObject type')
            if nested_object is self:
                raise ValueError(f'{repr(self)}: must not contain itself as nested object')

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'NyanObject<{self.name}>'

class NyanPatch(NyanObject):
    """
    Superclass for nyan patches.
    """
    __slots__ = ('_target', '_add_inheritance')

    def __init__(self, name: str, parents: OrderedSet[NyanObject]=None, members: OrderedSet[NyanObject]=None, nested_objects: OrderedSet[NyanObject]=None, target: NyanObject=None, add_inheritance: OrderedSet[NyanObject]=None):
        if False:
            while True:
                i = 10
        self._target = target
        self._add_inheritance = OrderedSet()
        if add_inheritance:
            self._add_inheritance.update(add_inheritance)
        super().__init__(name, parents, members, nested_objects)

    def get_target(self) -> NyanObject:
        if False:
            print('Hello World!')
        '\n        Returns the target of the patch.\n        '
        return self._target

    def is_abstract(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns True if unique or inherited members were\n        not initialized or the patch target is not set.\n        '
        return super().is_abstract() or not self._target

    @staticmethod
    def is_patch() -> bool:
        if False:
            print('Hello World!')
        '\n        Returns True if the object is a nyan patch.\n        '
        return True

    def set_target(self, target: NyanObject) -> NyanObject:
        if False:
            while True:
                i = 10
        '\n        Set the target of the patch.\n        '
        self._target = target
        if not isinstance(self._target, NyanObject):
            raise TypeError(f"{repr(self)}: '_target' must have NyanObject type")

    def dump(self, indent_depth: int=0, import_tree: ImportTree=None) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns the string representation of the object.\n        '
        output_str = f'{self.get_name()}'
        if import_tree:
            sfqon = '.'.join(import_tree.get_alias_fqon(self._target.get_fqon()))
        else:
            sfqon = '.'.join(self._target.get_fqon())
        output_str += f'<{sfqon}>'
        if len(self._add_inheritance) > 0:
            output_str += '['
            for new_inheritance in self._add_inheritance:
                if import_tree:
                    sfqon = '.'.join(import_tree.get_alias_fqon(new_inheritance.get_fqon()))
                else:
                    sfqon = '.'.join(new_inheritance.get_fqon())
                if new_inheritance[0] == 'FRONT':
                    output_str += f'+{sfqon}, '
                elif new_inheritance[0] == 'BACK':
                    output_str += f'{sfqon}+, '
            output_str = output_str[:-2] + ']'
        output_str += super()._prepare_inheritance_content(import_tree=import_tree)
        output_str += super()._prepare_object_content(indent_depth=indent_depth, import_tree=import_tree)
        return output_str

    def _sanity_check(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Check if the object conforms to nyan grammar rules. Also does\n        a bunch of type checks.\n        '
        super()._sanity_check()
        if self._target:
            if not isinstance(self._target, NyanObject):
                raise TypeError(f"{repr(self)}: '_target' must have NyanObject type")
        if len(self._add_inheritance) > 0:
            for inherit in self._add_inheritance:
                if not isinstance(inherit, tuple):
                    raise TypeError(f"{repr(self)}: '_add_inheritance' must be a tuple")
                if len(inherit) != 2:
                    raise SyntaxError(f"{repr(self)}: '_add_inheritance' tuples must have length 2")
                if inherit[0] not in ('FRONT', 'BACK'):
                    raise ValueError(f'{repr(self)}: added inheritance must be FRONT or BACK mode')
                if not isinstance(inherit[1], NyanObject):
                    raise ValueError(f'{repr(self)}: added inheritance must contain NyanObject')

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'NyanPatch<{self.name}<{self._target.name}>>'

class NyanMemberType:
    """
    Superclass for nyan member types.
    """
    __slots__ = ('_member_type', '_element_types')

    def __init__(self, member_type: typing.Union[str, MemberType, NyanObject], element_types: typing.Collection[NyanMemberType]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes the member type and does some correctness\n        checks, for your convenience.\n        '
        if isinstance(member_type, NyanObject):
            self._member_type = member_type
        else:
            self._member_type = MemberType(member_type)
        self._element_types = None
        if element_types:
            self._element_types = tuple(element_types)
        self._sanity_check()

    def get_type(self) -> MemberType:
        if False:
            print('Hello World!')
        '\n        Returns the member type.\n        '
        return self._member_type

    def get_real_type(self) -> MemberType:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the member type without wrapping modifiers.\n        '
        if self.is_modifier():
            return self._element_types[0].get_real_type()
        return self._member_type

    def get_element_types(self) -> tuple[NyanMemberType, ...]:
        if False:
            i = 10
            return i + 15
        '\n        Returns the element types.\n        '
        return self._element_types

    def get_real_element_types(self) -> tuple[NyanMemberType, ...]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the element types without wrapping modifiers.\n        '
        if self.is_modifier():
            return self._element_types[0].get_real_element_types()
        return self._element_types

    def is_primitive(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns True if the member type is a single value.\n        '
        return self._member_type in (MemberType.INT, MemberType.FLOAT, MemberType.TEXT, MemberType.FILE, MemberType.BOOLEAN)

    def is_real_primitive(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Returns True if the member type is a primitive wrapped in a modifier.\n        '
        if self.is_modifier():
            return self._element_types[0].is_real_primitive()
        return self.is_primitive()

    def is_complex(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if the member type is a collection.\n        '
        return self._member_type in (MemberType.SET, MemberType.ORDEREDSET, MemberType.DICT)

    def is_real_complex(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Returns True if the member type is a collection wrapped in a modifier.\n        '
        if self.is_modifier():
            return self._element_types[0].is_real_complex()
        return self.is_complex()

    def is_object(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if the member type is an object.\n        '
        return isinstance(self._member_type, NyanObject)

    def is_real_object(self) -> bool:
        if False:
            return 10
        '\n        Returns True if the member type is an object wrapped in a modifier.\n        '
        if self.is_modifier():
            return self._element_types[0].is_real_object()
        return self.is_object()

    def is_modifier(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Returns True if the member type is a modifier.\n        '
        return self._member_type in (MemberType.ABSTRACT, MemberType.CHILDREN, MemberType.OPTIONAL)

    def is_composite(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns True if the member is a composite type with at least one element type.\n        '
        return self.is_complex() or self.is_modifier()

    def accepts_op(self, operator: MemberOperator) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if an operator is compatible with the member type.\n        '
        if self.is_modifier():
            return self._element_types[0].accepts_op(operator)
        if self._member_type in (MemberType.INT, MemberType.FLOAT) and operator not in (MemberOperator.ASSIGN, MemberOperator.ADD, MemberOperator.SUBTRACT, MemberOperator.MULTIPLY, MemberOperator.DIVIDE):
            return False
        if self._member_type is MemberType.TEXT and operator not in (MemberOperator.ASSIGN, MemberOperator.ADD):
            return False
        if self._member_type is MemberType.FILE and operator is not MemberOperator.ASSIGN:
            return False
        if self._member_type is MemberType.BOOLEAN and operator not in (MemberOperator.ASSIGN, MemberOperator.AND, MemberOperator.OR):
            return False
        if self._member_type is MemberType.SET and operator not in (MemberOperator.ASSIGN, MemberOperator.ADD, MemberOperator.SUBTRACT, MemberOperator.AND, MemberOperator.OR):
            return False
        if self._member_type is MemberType.ORDEREDSET and operator not in (MemberOperator.ASSIGN, MemberOperator.ADD, MemberOperator.SUBTRACT, MemberOperator.AND, MemberOperator.OR):
            return False
        if self._member_type is MemberType.DICT and operator not in (MemberOperator.ASSIGN, MemberOperator.ADD, MemberOperator.SUBTRACT, MemberOperator.AND, MemberOperator.OR):
            return False
        return True

    def accepts_value(self, value) -> bool:
        if False:
            print('Hello World!')
        '\n        Check if a value is compatible with the member type.\n        '
        if value is MemberSpecialValue.NYAN_NONE:
            return self._member_type is MemberType.OPTIONAL
        if self.is_modifier():
            return self._element_types[0].accepts_value(value)
        if value is MemberSpecialValue.NYAN_INF and self._member_type not in (MemberType.INT, MemberType.FLOAT):
            return False
        if self.is_object():
            if not (value is self._member_type or value.has_ancestor(self._member_type)):
                return False
        return True

    def _sanity_check(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Check if the member type and element types are compatiable.\n        '
        if self.is_composite():
            if not self._element_types:
                raise TypeError(f'{repr(self)}: element types are required for composite types')
            if self.is_complex():
                for elem_type in self._element_types:
                    if elem_type.is_real_complex():
                        raise TypeError(f'{repr(self)}: element types cannot be complex but contains {elem_type}')
        elif self._element_types:
            raise TypeError(f'{repr(self)}: member type has element types but is not a composite')

    def dump(self, import_tree: ImportTree=None, namespace: tuple[str]=None) -> str:
        if False:
            print('Hello World!')
        '\n        Returns the nyan string representation of the member type.\n        '
        if self.is_primitive():
            return self._member_type.value
        if self.is_object():
            if import_tree:
                sfqon = '.'.join(import_tree.get_alias_fqon(self._member_type.get_fqon(), namespace))
            else:
                sfqon = '.'.join(self._member_type.get_fqon())
            return sfqon
        return f"{self._member_type.value}({', '.join((elem_type.dump(import_tree) for elem_type in self._element_types))})"

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'NyanMemberType<{self.dump()}>'

class NyanMember:
    """
    Superclass for all nyan members.
    """
    __slots__ = ('name', '_member_type', 'value', '_operator', '_override_depth')

    def __init__(self, name: str, member_type: NyanMemberType, value=None, operator: MemberOperator=None, override_depth: int=0):
        if False:
            i = 10
            return i + 15
        '\n        Initializes the member and does some correctness\n        checks, for your convenience.\n        '
        self.name = name
        if isinstance(member_type, NyanMemberType):
            self._member_type = member_type
        else:
            raise TypeError(f'NyanMember<{self.name}>: Expected NyanMemberType for member_type but got {type(member_type)}')
        self._override_depth = override_depth
        self._operator: MemberOperator = None
        if operator:
            operator = MemberOperator(operator)
        self.value = None
        if value is not None:
            self.set_value(value, operator)
        self._sanity_check()

    def get_name(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Returns the name of the member.\n        '
        return self.name

    def get_member_type(self) -> NyanMemberType:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the type of the member.\n        '
        return self._member_type

    def get_operator(self) -> MemberOperator:
        if False:
            print('Hello World!')
        '\n        Returns the operator of the member.\n        '
        return self._operator

    def get_override_depth(self) -> int:
        if False:
            return 10
        '\n        Returns the override depth of the member.\n        '
        return self._override_depth

    def get_value(self):
        if False:
            print('Hello World!')
        '\n        Returns the value of the member.\n        '
        return self.value

    def is_primitive(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns True if the member is a single value.\n        '
        return self._member_type.is_real_primitive()

    def is_complex(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns True if the member is a collection.\n        '
        return self._member_type.is_real_complex()

    def is_object(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if the member is an object.\n        '
        return self._member_type.is_real_object()

    def is_initialized(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns True if the member has a value.\n        '
        return self.value is not None

    @staticmethod
    def is_inherited() -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if the member is inherited from another object.\n        '
        return False

    def has_value(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns True if the member has a value.\n        '
        return self.value is not None

    def set_value(self, value, operator: MemberOperator=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set the value of the nyan member to the specified value and\n        optionally, the operator.\n        '
        if not self.value and (not operator):
            raise ValueError(f'Setting a value for an uninitialized member {repr(self)} requires also setting the operator')
        self.value = value
        self._operator = operator
        if self.value not in (MemberSpecialValue.NYAN_INF, MemberSpecialValue.NYAN_NONE):
            self._type_conversion()
        self._sanity_check()

    def dump(self, indent_depth: int, import_tree: ImportTree=None, namespace: tuple[str]=None) -> str:
        if False:
            return 10
        '\n        Returns the nyan string representation of the member.\n        '
        output_str = f'{self.name} : {self._member_type.dump(import_tree=import_tree)}'
        if self.is_initialized():
            value_str = self._get_value_str(indent_depth, import_tree=import_tree, namespace=namespace)
            output_str += f" {'@' * self._override_depth}{self._operator.value} {value_str}"
        return output_str

    def dump_short(self, indent_depth: int, import_tree: ImportTree=None, namespace: tuple[str]=None) -> str:
        if False:
            print('Hello World!')
        '\n        Returns the nyan string representation of the member, but\n        without the type definition.\n        '
        value_str = self._get_value_str(indent_depth, import_tree=import_tree, namespace=namespace)
        return f"{self.get_name()} {'@' * self._override_depth}{self._operator.value} {value_str}"

    def _sanity_check(self) -> None:
        if False:
            print('Hello World!')
        '\n        Check if the member conforms to nyan grammar rules. Also does\n        a bunch of type checks.\n        '
        if not isinstance(self.name, str):
            raise TypeError(f"{repr(self)}: 'name' must be a string")
        if not re.fullmatch('[a-zA-Z_][a-zA-Z0-9_]*', self.name[0]):
            raise SyntaxError(f"{repr(self)}: 'name' is not well-formed")
        if self.is_initialized() and (not self.is_inherited()) or (self.is_inherited() and self.has_value()):
            if not (isinstance(self._override_depth, int) and self._override_depth >= 0):
                raise ValueError(f'{repr(self)}: override depth must be a non-negative integer')
            if not self._member_type.accepts_op(self._operator):
                raise TypeError(f'{repr(self)}: {self._operator} is not a validoperator for member type {self._member_type}')
            if not self._member_type.accepts_value(self.value):
                raise TypeError(f"{repr(self)}: value '{self.value}' is not compatible with type '{self._member_type}'")

    def _type_conversion(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Explicit type conversion of the member value.\n\n        This lets us convert data fields without worrying about the\n        correct types too much, e.g. if a boolean is stored as uint8.\n        '
        if self._member_type.get_real_type() is MemberType.INT and self._operator not in (MemberOperator.DIVIDE, MemberOperator.MULTIPLY):
            self.value = int(self.value)
        elif self._member_type.get_real_type() is MemberType.FLOAT:
            self.value = float(self.value)
        elif self._member_type.get_real_type() is MemberType.TEXT:
            self.value = str(self.value)
        elif self._member_type.get_real_type() is MemberType.FILE:
            self.value = str(self.value)
        elif self._member_type.get_real_type() is MemberType.BOOLEAN:
            self.value = bool(self.value)
        elif self._member_type.get_real_type() is MemberType.SET:
            self.value = OrderedSet(self.value)
        elif self._member_type.get_real_type() is MemberType.ORDEREDSET:
            self.value = OrderedSet(self.value)
        elif self._member_type.get_real_type() is MemberType.DICT:
            self.value = dict(self.value)

    @staticmethod
    def _get_primitive_value_str(member_type: NyanMemberType, value, import_tree: ImportTree=None, namespace: tuple[str]=None) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Returns the nyan string representation of primitive values.\n\n        Subroutine of _get_value_str(..)\n        '
        if member_type.get_real_type() in (MemberType.TEXT, MemberType.FILE):
            return f'"{value}"'
        if member_type.is_real_object():
            if import_tree:
                sfqon = '.'.join(import_tree.get_alias_fqon(value.get_fqon(), namespace))
            else:
                sfqon = '.'.join(value.get_fqon())
            return sfqon
        return f'{value}'

    def _get_complex_value_str(self, indent_depth: int, member_type: NyanMemberType, value, import_tree: ImportTree=None, namespace: tuple[str]=None) -> str:
        if False:
            return 10
        '\n        Returns the nyan string representation of complex values.\n\n        Subroutine of _get_value_str()\n        '
        output_str = ''
        if member_type.get_real_type() is MemberType.ORDEREDSET:
            output_str += 'o'
        output_str += '{'
        stored_values = []
        if member_type.get_real_type() is MemberType.DICT:
            for (key, val) in value.items():
                subtype = member_type.get_real_element_types()[0]
                key_str = self._get_primitive_value_str(subtype, key, import_tree=import_tree, namespace=namespace)
                subtype = member_type.get_real_element_types()[1]
                val_str = self._get_primitive_value_str(subtype, val, import_tree=import_tree, namespace=namespace)
                stored_values.append(f'{key_str}: {val_str}')
        else:
            for val in value:
                subtype = member_type.get_real_element_types()[0]
                stored_values.append(self._get_primitive_value_str(subtype, val, import_tree=import_tree, namespace=namespace))
        concat_values = ', '.join(stored_values)
        line_length = len(indent_depth * INDENT) + len(f"{self.name} {'@' * self._override_depth}{self._operator.value} {concat_values}")
        if line_length < MAX_LINE_WIDTH:
            output_str += concat_values
        elif stored_values:
            output_str += '\n'
            space_left = MAX_LINE_WIDTH - len((indent_depth + 1) * INDENT)
            longest_len = len(max(stored_values, key=len))
            values_per_line = space_left // longest_len
            values_per_line = max(values_per_line, 1)
            output_str += (indent_depth + 1) * INDENT
            val_index = 0
            end_index = len(stored_values)
            for val in stored_values:
                val_index += 1
                output_str += val
                if val_index % values_per_line == 0:
                    output_str += ',\n'
                    if val_index != end_index:
                        output_str += (indent_depth + 1) * INDENT
                else:
                    output_str += ', '
            output_str = output_str[:-2] + '\n'
            output_str += indent_depth * INDENT
        output_str = output_str + '}'
        return output_str

    def _get_value_str(self, indent_depth: int, import_tree: ImportTree=None, namespace: tuple[str]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the nyan string representation of the value.\n        '
        if not self.is_initialized():
            return f'UNINITIALIZED VALUE {repr(self)}'
        if self.value is MemberSpecialValue.NYAN_NONE:
            return MemberSpecialValue.NYAN_NONE.value
        if self.value is MemberSpecialValue.NYAN_INF:
            return MemberSpecialValue.NYAN_INF.value
        if self.is_primitive() or self.is_object():
            return self._get_primitive_value_str(self._member_type, self.value, import_tree=import_tree, namespace=namespace)
        if self.is_complex():
            return self._get_complex_value_str(indent_depth, self._member_type, self.value, import_tree=import_tree, namespace=namespace)
        raise TypeError(f'{repr(self)} has no valid type')

    def __str__(self):
        if False:
            while True:
                i = 10
        return self._get_value_str(indent_depth=0)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'NyanMember<{self.name}: {self._member_type}>'

class NyanPatchMember(NyanMember):
    """
    Nyan members for patches.
    """
    __slots__ = ('_patch_target', '_member_origin')

    def __init__(self, name: str, patch_target: NyanObject, member_origin: NyanObject, value, operator: MemberOperator, override_depth: int=0):
        if False:
            return 10
        '\n        Initializes the member and does some correctness checks,\n        for your convenience. Other than the normal members,\n        patch members must initialize all values in the constructor\n        '
        self._patch_target = patch_target
        self._member_origin = member_origin
        target_member_type = self._get_target_member_type(name, member_origin)
        super().__init__(name, target_member_type, value, operator, override_depth)

    def get_name_with_origin(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the name of the member in <member_origin>.<name> form.\n        '
        return f'{self._member_origin.name}.{self.name}'

    def dump(self, indent_depth: int, import_tree: ImportTree=None, namespace: tuple[str]=None) -> str:
        if False:
            print('Hello World!')
        '\n        Returns the string representation of the member.\n        '
        return self.dump_short(indent_depth, import_tree=import_tree, namespace=namespace)

    def dump_short(self, indent_depth: int, import_tree: ImportTree=None, namespace: tuple[str]=None) -> str:
        if False:
            print('Hello World!')
        '\n        Returns the nyan string representation of the member, but\n        without the type definition.\n        '
        value_str = self._get_value_str(indent_depth, import_tree=import_tree, namespace=namespace)
        return f"{self.get_name_with_origin()} {'@' * self._override_depth}{self._operator.value} {value_str}"

    def _sanity_check(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Check if the member conforms to nyan grammar rules. Also does\n        a bunch of type checks.\n        '
        super()._sanity_check()
        if not isinstance(self._patch_target, NyanObject):
            raise TypeError(f"{self}: '_patch_target' must have NyanObject type")
        if not isinstance(self._member_origin, NyanObject):
            raise TypeError(f"{self}: '_member_origin' must have NyanObject type")

    def _get_target_member_type(self, name: str, origin: NyanObject):
        if False:
            i = 10
            return i + 15
        '\n        Retrieves the type of the patched member.\n        '
        target_member = self._member_origin.get_member_by_name(name, origin)
        return target_member.get_member_type()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'NyanPatchMember<{self.name}: {self._member_type}>'

class InheritedNyanMember(NyanMember):
    """
    Nyan members inherited from other objects.
    """
    __slots__ = ('_parent', '_origin')

    def __init__(self, name: str, member_type: NyanMemberType, parent: NyanObject, origin: NyanObject, value=None, operator: MemberOperator=None, override_depth: int=0):
        if False:
            i = 10
            return i + 15
        '\n        Initializes the member and does some correctness\n        checks, for your convenience.\n        '
        self._parent = parent
        self._origin = origin
        super().__init__(name, member_type, value, operator, override_depth)

    def get_name_with_origin(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns the name of the member in <origin>.<name> form.\n        '
        return f'{self._origin.name}.{self.name}'

    def get_origin(self) -> NyanObject:
        if False:
            print('Hello World!')
        '\n        Returns the origin of the member.\n        '
        return self._origin

    def get_parent(self) -> NyanObject:
        if False:
            while True:
                i = 10
        '\n        Returns the direct parent of the member.\n        '
        return self._parent

    @staticmethod
    def is_inherited() -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if the member is inherited from another object.\n        '
        return True

    def is_initialized(self) -> bool:
        if False:
            return 10
        '\n        Returns True if self or the parent is initialized.\n        '
        return super().is_initialized() or self._parent.get_member_by_name(self.name, self._origin).is_initialized()

    def dump(self, indent_depth: int, import_tree: ImportTree=None, namespace: tuple[str]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the string representation of the member.\n        '
        return self.dump_short(indent_depth, import_tree=import_tree, namespace=namespace)

    def dump_short(self, indent_depth: int, import_tree: ImportTree=None, namespace: tuple[str]=None) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Returns the nyan string representation of the member, but\n        without the type definition.\n        '
        value_str = self._get_value_str(indent_depth, import_tree=import_tree, namespace=namespace)
        return f"{self.get_name_with_origin()} {'@' * self._override_depth}{self._operator.value} {value_str}"

    def _sanity_check(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Check if the member conforms to nyan grammar rules. Also does\n        a bunch of type checks.\n        '
        super()._sanity_check()
        if not isinstance(self._parent, NyanObject):
            raise TypeError(f"{repr(self)}: '_parent' must have NyanObject type")
        if not isinstance(self._origin, NyanObject):
            raise TypeError(f"{repr(self)}: '_origin' must have NyanObject type")

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'InheritedNyanMember<{self.name}: {self._member_type}>'

class MemberType(Enum):
    """
    Symbols for nyan member types.
    """
    INT = 'int'
    FLOAT = 'float'
    TEXT = 'text'
    FILE = 'file'
    BOOLEAN = 'bool'
    SET = 'set'
    ORDEREDSET = 'orderedset'
    DICT = 'dict'
    ABSTRACT = 'abstract'
    CHILDREN = 'children'
    OPTIONAL = 'optional'

class MemberSpecialValue(Enum):
    """
    Symbols for special nyan values.
    """
    NYAN_NONE = 'None'
    NYAN_INF = 'inf'

class MemberOperator(Enum):
    """
    Symbols for nyan member operators.
    """
    ASSIGN = '='
    ADD = '+='
    SUBTRACT = '-='
    MULTIPLY = '*='
    DIVIDE = '/='
    AND = '&='
    OR = '|='