"""
Objects that represent data structures in the original game.

These are simple containers that can be processed by the converter.
"""
from __future__ import annotations
import typing
from openage.convert.value_object.read.dynamic_loader import DynamicLoader
from ....nyan.nyan_structs import NyanObject, NyanPatch, NyanPatchMember, MemberOperator
from ...value_object.conversion.forward_ref import ForwardRef
from ...value_object.read.value_members import NoDiffMember, ValueMember
from .combined_sound import CombinedSound
from .combined_sprite import CombinedSprite
from .combined_terrain import CombinedTerrain

class ConverterObject:
    """
    Storage object for data objects in the to-be-converted games.
    """
    __slots__ = ('obj_id', 'members')

    def __init__(self, obj_id: typing.Union[str, int], members: dict[str, ValueMember]=None):
        if False:
            print('Hello World!')
        '\n        Creates a new ConverterObject.\n\n        :param obj_id: An identifier for the object (as a string or int)\n        :type obj_id: str|int\n        :param members: An already existing member dict.\n        :type members: dict[str, ValueMember]\n        '
        self.obj_id = obj_id
        self.members = {}
        if members:
            if isinstance(members, DynamicLoader):
                self.members = members
            elif all((isinstance(member, ValueMember) for member in members.values())):
                self.members.update(members)
            else:
                raise TypeError('members must be an instance of ValueMember')

    def get_id(self) -> typing.Union[str, int]:
        if False:
            i = 10
            return i + 15
        "\n        Returns the object's ID.\n        "
        return self.obj_id

    def add_member(self, member: ValueMember) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Adds a member to the object.\n        '
        self.members.update({member.name: member})

    def add_members(self, members: dict[str, ValueMember]) -> None:
        if False:
            while True:
                i = 10
        '\n        Adds multiple members to the object.\n        '
        self.members.update(members)

    def get_member(self, member_id: str) -> ValueMember:
        if False:
            while True:
                i = 10
        '\n        Returns a member of the object.\n        '
        try:
            return self.members[member_id]
        except KeyError as err:
            raise KeyError(f'{self} has no attribute: {member_id}') from err

    def has_member(self, member_id: str) -> bool:
        if False:
            return 10
        '\n        Returns True if the object has a member with the specified name.\n        '
        return member_id in self.members

    def remove_member(self, member_id: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Removes a member from the object.\n        '
        self.members.pop(member_id, None)

    def short_diff(self, other: ConverterObject) -> ConverterObject:
        if False:
            print('Hello World!')
        '\n        Returns the obj_diff between two objects as another ConverterObject.\n\n        The object created by short_diff() only contains members\n        that are different. It does not contain NoDiffMembers.\n        '
        if type(self) is not type(other):
            raise TypeError(f'type {type(self)} cannot be diffed with type {type(other)}')
        obj_diff = {}
        for (member_id, member) in self.members.items():
            member_diff = member.diff(other.get_member(member_id))
            if not isinstance(member_diff, NoDiffMember):
                obj_diff.update({member_id: member_diff})
        return ConverterObject(f'{self.obj_id}-{other.get_id()}-sdiff', members=obj_diff)

    def diff(self, other: ConverterObject) -> ConverterObject:
        if False:
            while True:
                i = 10
        '\n        Returns the obj_diff between two objects as another ConverterObject.\n        '
        if type(self) is not type(other):
            raise TypeError(f'type {type(self)} cannot be diffed with type {type(other)}')
        obj_diff = {}
        for (member_id, member) in self.members.items():
            obj_diff.update({member_id: member.diff(other.get_member(member_id))})
        return ConverterObject(f'{self.obj_id}-{other.get_id()}-diff', members=obj_diff)

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        '\n        Short command for getting a member of the object.\n        '
        return self.get_member(key)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError(f'return short description of the object {type(self)}')

class ConverterObjectGroup:
    """
    A group of objects that are connected together in some way
    and need each other for conversion. ConverterObjectGroup
    instances are converted to the nyan API.
    """
    __slots__ = ('group_id', 'raw_api_objects', 'raw_member_pushs')

    def __init__(self, group_id: typing.Union[str, int], raw_api_objects: list[RawAPIObject]=None):
        if False:
            return 10
        '\n        Creates a new ConverterObjectGroup.\n\n        :paran group_id:  An identifier for the object group (as a string or int)\n        :param raw_api_objects: A list of raw API objects. These will become\n                                proper API objects during conversion.\n        '
        self.group_id = group_id
        self.raw_api_objects = {}
        self.raw_member_pushs = []
        if raw_api_objects:
            self._create_raw_api_object_dict(raw_api_objects)

    def get_id(self) -> typing.Union[str, int]:
        if False:
            print('Hello World!')
        "\n        Returns the object group's ID.\n        "
        return self.group_id

    def add_raw_api_object(self, subobject: RawAPIObject) -> None:
        if False:
            while True:
                i = 10
        '\n        Adds a subobject to the object.\n        '
        key = subobject.get_id()
        self.raw_api_objects.update({key: subobject})

    def add_raw_api_objects(self, subobjects: list[RawAPIObject]) -> None:
        if False:
            while True:
                i = 10
        '\n        Adds several subobject to the object.\n        '
        for subobject in subobjects:
            self.add_raw_api_object(subobject)

    def add_raw_member_push(self, push_object: RawMemberPush) -> None:
        if False:
            print('Hello World!')
        '\n        Adds a RawPushMember to the object.\n        '
        self.raw_member_pushs.append(push_object)

    def create_nyan_objects(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates nyan objects from the existing raw API objects.\n        '
        patch_objects = []
        for raw_api_object in self.raw_api_objects.values():
            raw_api_object.create_nyan_object()
            if raw_api_object.is_patch():
                patch_objects.append(raw_api_object)
        for patch_object in patch_objects:
            patch_object.link_patch_target()

    def create_nyan_members(self) -> None:
        if False:
            print('Hello World!')
        '\n        Fill nyan members of all raw API objects.\n        '
        for raw_api_object in self.raw_api_objects.values():
            raw_api_object.create_nyan_members()

    def check_readiness(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        check if all nyan objects in the group are ready for export.\n        '
        for raw_api_object in self.raw_api_objects.values():
            if not raw_api_object.is_ready():
                if not raw_api_object.nyan_object:
                    raise ValueError(f'{raw_api_object}: object is not ready for export: Nyan object not initialized.')
                uninit_members = raw_api_object.get_nyan_object().get_uninitialized_members()
                concat_names = ', '.join((f"'{member.get_name()}'" for member in uninit_members))
                raise ValueError(f'{raw_api_object}: object is not ready for export: Member(s) {concat_names} not initialized.')

    def execute_raw_member_pushs(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Extend raw members of referenced raw API objects.\n        '
        for push_object in self.raw_member_pushs:
            forward_ref = push_object.get_object_target()
            raw_api_object = forward_ref.resolve_raw()
            raw_api_object.extend_raw_member(push_object.get_member_name(), push_object.get_push_value(), push_object.get_member_origin())

    def get_raw_api_object(self, obj_id: str) -> RawAPIObject:
        if False:
            i = 10
            return i + 15
        '\n        Returns a subobject of the object.\n        '
        try:
            return self.raw_api_objects[obj_id]
        except KeyError as missing_raw_api_obj:
            raise KeyError(f'{repr(self)}: Could not find raw API object with obj_id {obj_id}') from missing_raw_api_obj

    def get_raw_api_objects(self) -> dict[str, RawAPIObject]:
        if False:
            print('Hello World!')
        '\n        Returns all raw API objects.\n        '
        return self.raw_api_objects

    def has_raw_api_object(self, obj_id: typing.Union[str, int]) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns True if the object has a subobject with the specified ID.\n        '
        return obj_id in self.raw_api_objects

    def remove_raw_api_object(self, obj_id: typing.Union[str, int]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Removes a subobject from the object.\n        '
        self.raw_api_objects.pop(obj_id)

    def _create_raw_api_object_dict(self, subobject_list: list[RawAPIObject]) -> None:
        if False:
            while True:
                i = 10
        '\n        Creates the dict from the subobject list passed to __init__.\n        '
        for subobject in subobject_list:
            self.add_raw_api_object(subobject)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'ConverterObjectGroup<{self.group_id}>'

class RawAPIObject:
    """
    An object that contains all the necessary information to create
    a nyan API object. Members are stored as (membername, value) pairs.
    Values refer either to primitive values (int, float, str),
    forward references to objects or expected media files.
    The 'expected' values two have to be resolved in an additional step.
    """
    __slots__ = ('obj_id', 'name', 'api_ref', 'raw_members', 'raw_parents', '_location', '_filename', 'nyan_object', '_patch_target', 'raw_patch_parents')

    def __init__(self, obj_id: typing.Union[str, int], name: str, api_ref: dict[str, NyanObject], location: typing.Union[str, ForwardRef]=''):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a raw API object.\n\n        :param obj_id: Unique identifier for the raw API object.\n        :type obj_id: str\n        :param name: Name of the nyan object created from the raw API object.\n        :type name: str\n        :param api_ref: The openage API objects used as reference for creating the nyan object.\n        :type api_ref: dict\n        :param location: Relative path of the nyan file in the modpack or another raw API object.\n        :type location: str, .forward_ref.ForwardRef\n        '
        self.obj_id = obj_id
        self.name = name
        self.api_ref = api_ref
        self.raw_members = []
        self.raw_parents = []
        self.raw_patch_parents = []
        self._location = location
        self._filename = None
        self.nyan_object = None
        self._patch_target = None

    def add_raw_member(self, name: str, value: typing.Union[int, float, bool, str, list, dict, ForwardRef], origin: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a raw member to the object.\n\n        :param name: Name of the member (has to be a valid inherited member name).\n        :type name: str\n        :param value: Value of the member.\n        :type value: int, float, bool, str, list, dict, ForwardRef\n        :param origin: fqon of the object from which the member was inherited.\n        :type origin: str\n        '
        self.raw_members.append((name, value, origin))

    def add_raw_patch_member(self, name: str, value: typing.Union[int, float, bool, str, list, dict, ForwardRef], origin: str, operator: MemberOperator) -> None:
        if False:
            while True:
                i = 10
        '\n        Adds a raw patch member to the object.\n\n        :param name: Name of the member (has to be a valid target member name).\n        :type name: str\n        :param value: Value of the member.\n        :type value: int, float, bool, str, list, dict, ForwardRef\n        :param origin: fqon of the object from which the member was inherited.\n        :type origin: str\n        :param operator: the operator for the patched member\n        :type operator: MemberOperator\n        '
        self.raw_members.append((name, value, origin, operator))

    def add_raw_parent(self, parent_id: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Adds a raw parent to the object.\n\n        :param parent_id: fqon of the parent in the API object dictionary\n        :type parent_id: str\n        '
        self.raw_parents.append(parent_id)

    def add_raw_patch_parent(self, parent_id: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Adds a raw patch parent to the object.\n\n        :param parent_id: fqon of the parent in the API object dictionary\n        :type parent_id: str\n        '
        self.raw_patch_parents.append(parent_id)

    def extend_raw_member(self, name: str, push_value: list, origin: str) -> None:
        if False:
            print('Hello World!')
        '\n        Extends a raw member value if the value is a list.\n\n        :param name: Name of the member (has to be a valid inherited member name).\n        :type name: str\n        :param push_value: Extended value of the member.\n        :type push_value: list\n        :param origin: fqon of the object from which the member was inherited.\n        :type origin: str\n        '
        for raw_member in self.raw_members:
            member_name = raw_member[0]
            member_value = raw_member[1]
            member_origin = raw_member[2]
            if name == member_name and member_origin == origin:
                member_value.extend(push_value)
                break
        else:
            raise ValueError(f'{repr(self)}: Cannot extend raw member {name} with origin {origin}: member not found')

    def create_nyan_object(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Create the nyan object for this raw API object. Members have to be created separately.\n        '
        parents = []
        for raw_parent in self.raw_parents:
            parents.append(self.api_ref[raw_parent])
        if self.is_patch():
            self.nyan_object = NyanPatch(self.name, parents)
        else:
            self.nyan_object = NyanObject(self.name, parents)

    def create_nyan_members(self) -> None:
        if False:
            return 10
        '\n        Fills the nyan object members with values from the raw members.\n        References to nyan objects or media files with be resolved.\n        The nyan object has to be created before this function can be called.\n        '
        if self.nyan_object is None:
            raise RuntimeError(f'{repr(self)}: nyan object needs to be created before member values can be assigned')
        for raw_member in self.raw_members:
            member_name = raw_member[0]
            member_value = raw_member[1]
            member_origin = self.api_ref[raw_member[2]]
            member_operator = None
            if self.is_patch():
                member_operator = raw_member[3]
            member_value = self._resolve_raw_values(member_value)
            if self.is_patch():
                nyan_member = NyanPatchMember(member_name, self.nyan_object.get_target(), member_origin, member_value, member_operator)
                self.nyan_object.add_member(nyan_member)
            else:
                nyan_member = self.nyan_object.get_member_by_name(member_name, member_origin)
                nyan_member.set_value(member_value, MemberOperator.ASSIGN)

    def link_patch_target(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set the target NyanObject for a patch.\n        '
        if not self.is_patch():
            raise TypeError(f'Cannot link patch target: {self} is not a patch')
        if isinstance(self._patch_target, ForwardRef):
            target = self._patch_target.resolve()
        else:
            target = self._patch_target
        self.nyan_object.set_target(target)

    def get_filename(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Returns the filename of the raw API object.\n        '
        return self._filename

    def get_file_location(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Returns a tuple with\n            1. the relative path to the directory\n            2. the filename\n        where the nyan object will be stored.\n\n        This method can be called instead of get_location() when\n        you are unsure whether the nyan object will be nested.\n        '
        if isinstance(self._location, ForwardRef):
            nesting_raw_api_object = self._location.resolve_raw()
            nesting_location = nesting_raw_api_object.get_location()
            while isinstance(nesting_location, ForwardRef):
                nesting_raw_api_object = nesting_location.resolve_raw()
                nesting_location = nesting_raw_api_object.get_location()
            return (nesting_location, nesting_raw_api_object.get_filename())
        return (self._location, self._filename)

    def get_id(self) -> typing.Union[str, int]:
        if False:
            while True:
                i = 10
        '\n        Returns the ID of the raw API object.\n        '
        return self.obj_id

    def get_location(self) -> typing.Union[str, ForwardRef]:
        if False:
            i = 10
            return i + 15
        '\n        Returns the relative path to a directory or an ForwardRef\n        to another RawAPIObject.\n        '
        return self._location

    def get_nyan_object(self) -> NyanObject:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the nyan API object for the raw API object.\n        '
        if self.nyan_object:
            return self.nyan_object
        raise RuntimeError(f'nyan object for {self} has not been created yet')

    def is_ready(self) -> bool:
        if False:
            return 10
        '\n        Returns whether the object is ready to be exported.\n        '
        return self.nyan_object is not None and (not self.nyan_object.is_abstract())

    def is_patch(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Returns True if the object is a patch.\n        '
        return self._patch_target is not None

    def set_filename(self, filename: str, suffix: str='nyan') -> None:
        if False:
            print('Hello World!')
        '\n        Set the filename of the resulting nyan file.\n\n        :param filename: File name prefix (without extension).\n        :type filename: str\n        :param suffix: File extension (defaults to "nyan")\n        :type suffix: str\n        '
        self._filename = f'{filename}.{suffix}'

    def set_location(self, location: typing.Union[str, ForwardRef]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the relative location of the object in a modpack. This must\n        be a path to a nyan file or an ForwardRef to a nyan object.\n\n        :param location: Relative path of the nyan file in the modpack or\n                         a forward reference to another raw API object.\n        :type location: str, ForwardRef\n        '
        self._location = location

    def set_patch_target(self, target: typing.Union[ForwardRef, NyanObject]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set an ForwardRef as a target for this object. If this\n        is done, the RawAPIObject will be converted to a patch.\n\n        :param target: A forward reference to another raw API object or a nyan object.\n        :type target: ForwardRef, NyanObject\n        '
        self._patch_target = target

    @staticmethod
    def _resolve_raw_value(value) -> typing.Union[NyanObject, str, float]:
        if False:
            while True:
                i = 10
        '\n        Check if a raw member value contains a reference to a resource (nyan\n        objects or asset files), resolve the reference to a nyan-compatible value\n        and return it.\n\n        If the value contains no resource reference, it is returned as-is.\n\n        :param value: Raw member value.\n        :return: Value usable by a nyan object or nyan member.\n        '
        if isinstance(value, ForwardRef):
            return value.resolve()
        if isinstance(value, CombinedSprite):
            return value.get_relative_sprite_location()
        if isinstance(value, CombinedTerrain):
            return value.get_relative_terrain_location()
        if isinstance(value, CombinedSound):
            return value.get_relative_file_location()
        if isinstance(value, float):
            return round(value, ndigits=6)
        return value

    def _resolve_raw_values(self, values):
        if False:
            print('Hello World!')
        '\n        Convert a raw member values to nyan-compatible values by resolving\n        contained references to resources.\n\n        :param values: Raw member values.\n        :return: Value usable by a nyan object or nyan member.\n        '
        if isinstance(values, list):
            temp_values = []
            for temp_value in values:
                temp_values.append(self._resolve_raw_value(temp_value))
            return temp_values
        if isinstance(values, dict):
            temp_values = {}
            for (key, val) in values.items():
                temp_values.update({self._resolve_raw_value(key): self._resolve_raw_value(val)})
            return temp_values
        return self._resolve_raw_value(values)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'RawAPIObject<{self.obj_id}>'

class RawMemberPush:
    """
    An object that contains additional values for complex members
    in raw API objects (lists or sets). Pushing these values to the
    raw API object will extennd the list or set. The values should be
    pushed to the raw API objects before their nyan members are created.
    """
    __slots__ = ('forward_ref', 'member_name', 'member_origin', 'push_value')

    def __init__(self, forward_ref: ForwardRef, member_name: str, member_origin: str, push_value: list):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new member push.\n\n        :param forward_ref: forward reference of the RawAPIObject.\n        :type forward_ref: ForwardRef\n        :param member_name: Name of the member that is extended.\n        :type member_name: str\n        :param member_origin: Fqon of the object the member was inherited from.\n        :type member_origin: str\n        :param push_value: Value that extends the existing member value.\n        :type push_value: list\n        '
        self.forward_ref = forward_ref
        self.member_name = member_name
        self.member_origin = member_origin
        self.push_value = push_value

    def get_object_target(self) -> ForwardRef:
        if False:
            i = 10
            return i + 15
        '\n        Returns the forward reference for the push target.\n        '
        return self.forward_ref

    def get_member_name(self) -> str:
        if False:
            return 10
        '\n        Returns the name of the member that is extended.\n        '
        return self.member_name

    def get_member_origin(self) -> str:
        if False:
            i = 10
            return i + 15
        "\n        Returns the fqon of the member's origin.\n        "
        return self.member_origin

    def get_push_value(self) -> list:
        if False:
            return 10
        "\n        Returns the value that extends the member's existing value.\n        "
        return self.push_value

class ConverterObjectContainer:
    """
    A conainer for all ConverterObject instances in a converter process.

    It is recommended to create one ConverterObjectContainer for everything
    and pass the reference around.
    """

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'ConverterObjectContainer'