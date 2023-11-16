from __future__ import annotations
import typing
import math
import re
import struct
from openage.convert.value_object.read.dynamic_loader import DynamicLoader
from ....util.strings import decode_until_null
from .member_access import READ, READ_GEN, READ_UNKNOWN, SKIP, MemberAccess
from .read_members import IncludeMembers, ContinueReadMember, MultisubtypeMember, GroupMember, SubdataMember, ReadMember, EnumLookupMember
from .value_members import ContainerMember, ArrayMember, IntMember, FloatMember, StringMember, BooleanMember, IDMember, BitfieldMember, ValueMember
from .value_members import StorageType
if typing.TYPE_CHECKING:
    from openage.convert.value_object.init.game_version import GameVersion
VARARRAY_MATCH = re.compile('([{0}]+) *\\[([{0}]+)\\] *;?'.format('a-zA-Z0-9_'))
INTEGER_MATCH = re.compile('\\d+')
STRUCT_TYPE_LOOKUP = {'char': 'b', 'unsigned char': 'B', 'int8_t': 'b', 'uint8_t': 'B', 'short': 'h', 'unsigned short': 'H', 'int16_t': 'h', 'uint16_t': 'H', 'int': 'i', 'unsigned int': 'I', 'int32_t': 'i', 'uint32_t': 'I', 'long': 'l', 'unsigned long': 'L', 'long long': 'q', 'unsigned long long': 'Q', 'int64_t': 'q', 'uint64_t': 'Q', 'float': 'f', 'double': 'd', 'char[]': 's'}

class GenieStructure:
    """
    superclass for all structures from Genie Engine games.
    """
    dynamic_load = False

    def __init__(self, **args):
        if False:
            print('Hello World!')
        self.__dict__.update(args)

    def read(self, raw: bytes, offset: int, game_version: GameVersion, cls: GenieStructure=None, members: tuple=None, dynamic_load=False) -> tuple[int, list[ValueMember]]:
        if False:
            print('Hello World!')
        '\n        recursively read defined binary data from raw at given offset.\n\n        this is used to fill the python classes with data from the binary input.\n        '
        if cls:
            target_class = cls
        else:
            target_class = self
        generated_value_members = []
        stop_reading_members = False
        if not members:
            members = target_class.get_data_format(game_version, allowed_modes=(True, READ, READ_GEN, READ_UNKNOWN, SKIP), flatten_includes=False)
        start_offset = offset
        for (_, export, var_name, storage_type, var_type) in members:
            if export == READ_GEN and dynamic_load and self.dynamic_load:
                export = READ
            if stop_reading_members:
                if isinstance(var_type, ReadMember):
                    replacement_value = var_type.get_empty_value()
                else:
                    replacement_value = 0
                setattr(self, var_name, replacement_value)
                continue
            if isinstance(var_type, GroupMember):
                (offset, gen_members) = self._read_group(raw, offset, game_version, export, var_name, storage_type, var_type)
                generated_value_members.extend(gen_members)
            elif isinstance(var_type, MultisubtypeMember):
                (offset, gen_members) = self._read_multisubtye(raw, offset, game_version, export, var_name, storage_type, var_type, target_class)
                generated_value_members.extend(gen_members)
            else:
                (offset, gen_members, stop_reading_members) = self._read_primitive(raw, offset, export, var_name, storage_type, var_type)
                generated_value_members.extend(gen_members)
        if dynamic_load and self.dynamic_load:
            return (offset, DynamicLoader('', self.__class__, game_version, raw, start_offset))
        return (offset, generated_value_members)

    def _read_group(self, raw: bytes, offset: int, game_version: GameVersion, export: MemberAccess, var_name: str, storage_type: StorageType, var_type: GroupMember) -> tuple[int, list[ValueMember]]:
        if False:
            return 10
        generated_value_members = []
        if not issubclass(var_type.cls, GenieStructure):
            raise TypeError('class where members should be included is not exportable: %s' % var_type.cls.__name__)
        if isinstance(var_type, IncludeMembers):
            (offset, gen_members) = var_type.cls.read(self, raw, offset, game_version, cls=var_type.cls)
            if export == READ_GEN:
                generated_value_members.extend(gen_members)
        else:
            grouped_data = var_type.cls()
            (offset, gen_members) = grouped_data.read(raw, offset, game_version)
            setattr(self, var_name, grouped_data)
            if export == READ_GEN:
                if storage_type is StorageType.CONTAINER_MEMBER:
                    container = ContainerMember(var_name, gen_members)
                    generated_value_members.append(container)
                elif storage_type is StorageType.ARRAY_CONTAINER:
                    container = ContainerMember('', gen_members)
                    allowed_member_type = StorageType.CONTAINER_MEMBER
                    array = ArrayMember(var_name, allowed_member_type, [container])
                    generated_value_members.append(array)
                else:
                    raise SyntaxError('%s at offset %# 08x: Data read via %s cannot be stored as %s; expected %s or %s' % (var_name, offset, var_type, storage_type, StorageType.CONTAINER_MEMBER, StorageType.ARRAY_CONTAINER))
        return (offset, generated_value_members)

    def _read_multisubtye(self, raw: bytes, offset: int, game_version: GameVersion, export: MemberAccess, var_name: str, storage_type: StorageType, var_type: GroupMember, target_class: type) -> tuple[int, list[ValueMember]]:
        if False:
            return 10
        generated_value_members = []
        varargs = dict()
        if var_type.passed_args:
            if isinstance(var_type.passed_args, str):
                var_type.passed_args = set(var_type.passed_args)
            for passed_member_name in var_type.passed_args:
                varargs[passed_member_name] = getattr(self, passed_member_name)
        list_len = var_type.get_length(self)
        if isinstance(var_type, SubdataMember):
            setattr(self, var_name, list())
            single_type_subdata = True
        else:
            setattr(self, var_name, {key: [] for key in var_type.class_lookup})
            single_type_subdata = False
        subdata_value_members = []
        allowed_member_type = StorageType.CONTAINER_MEMBER
        if var_type.offset_to:
            offset_lookup = getattr(self, var_type.offset_to[0])
        else:
            offset_lookup = None
        for i in range(list_len):
            sub_members = []
            if offset_lookup:
                if not var_type.offset_to[1](offset_lookup[i]):
                    continue
            if single_type_subdata:
                new_data_class = var_type.class_lookup[None]
            else:
                (offset, sub_members) = self.read(raw, offset, game_version, cls=target_class, members=((False,) + var_type.subtype_definition,))
                subtype_name = getattr(self, var_type.subtype_definition[1])
                new_data_class = var_type.class_lookup[subtype_name]
            if not issubclass(new_data_class, GenieStructure):
                raise TypeError('dumped data is not exportable: %s' % new_data_class.__name__)
            new_data = new_data_class(**varargs)
            (offset, gen_members) = new_data.read(raw, offset, game_version, new_data_class)
            if single_type_subdata:
                getattr(self, var_name).append(new_data)
            else:
                getattr(self, var_name)[subtype_name].append(new_data)
            if export == READ_GEN:
                if storage_type is StorageType.ARRAY_CONTAINER:
                    sub_members.extend(gen_members)
                    gen_members = sub_members
                    container = ContainerMember('', gen_members)
                    subdata_value_members.append(container)
                else:
                    raise SyntaxError('%s at offset %# 08x: Data read via %s cannot be stored as %s; expected %s' % (var_name, offset, var_type, storage_type, StorageType.ARRAY_CONTAINER))
        if export == READ_GEN:
            array = ArrayMember(var_name, allowed_member_type, subdata_value_members)
            generated_value_members.append(array)
        return (offset, generated_value_members)

    def _read_primitive(self, raw: bytes, offset: int, export: MemberAccess, var_name: str, storage_type: StorageType, var_type: GroupMember) -> tuple[int, list[ValueMember], bool]:
        if False:
            return 10
        generated_value_members = []
        stop_reading_members = False
        data_count = 1
        is_array = False
        is_custom_member = False
        if isinstance(var_type, str):
            is_array = VARARRAY_MATCH.match(var_type)
            if is_array:
                struct_type = is_array.group(1)
                data_count = is_array.group(2)
                if struct_type == 'char':
                    struct_type = 'char[]'
                if INTEGER_MATCH.match(data_count):
                    data_count = int(data_count)
                else:
                    data_count = getattr(self, data_count)
                if storage_type not in (StorageType.STRING_MEMBER, StorageType.ARRAY_INT, StorageType.ARRAY_FLOAT, StorageType.ARRAY_BOOL, StorageType.ARRAY_ID, StorageType.ARRAY_STRING):
                    raise SyntaxError('%s at offset %# 08x: Data read via %s cannot be stored as %s; expected ArrayMember format' % (var_name, offset, var_type, storage_type))
            else:
                struct_type = var_type
                data_count = 1
        elif isinstance(var_type, ReadMember):
            struct_type = var_type.raw_type
            data_count = var_type.get_length(self)
            is_custom_member = True
        else:
            raise TypeError(f"unknown data member definition {var_type} for member '{var_name}'")
        if data_count < 0:
            raise SyntaxError("invalid length %d < 0 in %s for member '%s'" % (data_count, var_type, var_name))
        if struct_type not in STRUCT_TYPE_LOOKUP:
            raise TypeError('%s: member %s requests unknown data type %s' % (repr(self), var_name, struct_type))
        if export == READ_UNKNOWN:
            var_name = 'unknown-0x%08x' % offset
        symbol = STRUCT_TYPE_LOOKUP[struct_type]
        struct_format = '< %d%s' % (data_count, symbol)
        if export != SKIP:
            result = struct.unpack_from(struct_format, raw, offset)
            if is_custom_member:
                if not var_type.verify_read_data(self, result):
                    raise SyntaxError('invalid data when reading %s at offset %# 08x' % (var_name, offset))
            if symbol == 's':
                result = decode_until_null(result[0])
                if export == READ_GEN:
                    if storage_type is StorageType.STRING_MEMBER:
                        gen_member = StringMember(var_name, result)
                    else:
                        raise SyntaxError('%s at offset %# 08x: Data read via %s cannot be stored as %s; expected %s' % (var_name, offset, var_type, storage_type, StorageType.STRING_MEMBER))
                    generated_value_members.append(gen_member)
            elif is_array:
                if export == READ_GEN:
                    array_members = []
                    allowed_member_type = None
                    for elem in result:
                        if storage_type is StorageType.ARRAY_INT:
                            gen_member = IntMember(var_name, elem)
                            allowed_member_type = StorageType.INT_MEMBER
                            array_members.append(gen_member)
                        elif storage_type is StorageType.ARRAY_FLOAT:
                            gen_member = FloatMember(var_name, elem)
                            allowed_member_type = StorageType.FLOAT_MEMBER
                            array_members.append(gen_member)
                        elif storage_type is StorageType.ARRAY_BOOL:
                            gen_member = BooleanMember(var_name, elem)
                            allowed_member_type = StorageType.BOOLEAN_MEMBER
                            array_members.append(gen_member)
                        elif storage_type is StorageType.ARRAY_ID:
                            gen_member = IDMember(var_name, elem)
                            allowed_member_type = StorageType.ID_MEMBER
                            array_members.append(gen_member)
                        elif storage_type is StorageType.ARRAY_STRING:
                            gen_member = StringMember(var_name, elem)
                            allowed_member_type = StorageType.STRING_MEMBER
                            array_members.append(gen_member)
                        else:
                            raise SyntaxError('%s at offset %# 08x: Data read via %s cannot be stored as %s; expected %s, %s, %s, %s or %s' % (var_name, offset, var_type, storage_type, StorageType.ARRAY_INT, StorageType.ARRAY_FLOAT, StorageType.ARRAY_BOOL, StorageType.ARRAY_ID, StorageType.ARRAY_STRING))
                    array = ArrayMember(var_name, allowed_member_type, array_members)
                    generated_value_members.append(array)
            elif data_count == 1:
                result = result[0]
                if symbol == 'f':
                    if not math.isfinite(result):
                        raise SyntaxError('invalid float when reading %s at offset %# 08x' % (var_name, offset))
                if export == READ_GEN:
                    if is_custom_member:
                        lookup_result = var_type.entry_hook(result)
                        if isinstance(var_type, EnumLookupMember):
                            if storage_type is StorageType.INT_MEMBER:
                                gen_member = IntMember(var_name, result)
                            elif storage_type is StorageType.ID_MEMBER:
                                gen_member = IDMember(var_name, result)
                            elif storage_type is StorageType.BITFIELD_MEMBER:
                                gen_member = BitfieldMember(var_name, result)
                            elif storage_type is StorageType.STRING_MEMBER:
                                gen_member = StringMember(var_name, lookup_result)
                            else:
                                raise SyntaxError('%s at offset %# 08x: Data read via %s cannot be stored as %s; expected %s, %s, %s or %s' % (var_name, offset, var_type, storage_type, StorageType.INT_MEMBER, StorageType.ID_MEMBER, StorageType.BITFIELD_MEMBER, StorageType.STRING_MEMBER))
                        elif isinstance(var_type, ContinueReadMember):
                            if storage_type is StorageType.BOOLEAN_MEMBER:
                                gen_member = StringMember(var_name, lookup_result)
                            else:
                                raise SyntaxError('%s at offset %# 08x: Data read via %s cannot be stored as %s; expected %s' % (var_name, offset, var_type, storage_type, StorageType.BOOLEAN_MEMBER))
                    elif storage_type is StorageType.INT_MEMBER:
                        gen_member = IntMember(var_name, result)
                    elif storage_type is StorageType.FLOAT_MEMBER:
                        gen_member = FloatMember(var_name, result)
                    elif storage_type is StorageType.BOOLEAN_MEMBER:
                        gen_member = BooleanMember(var_name, result)
                    elif storage_type is StorageType.ID_MEMBER:
                        gen_member = IDMember(var_name, result)
                    else:
                        raise SyntaxError('%s at offset %# 08x: Data read via %s cannot be stored as %s; expected %s, %s, %s or %s' % (var_name, offset, var_type, storage_type, StorageType.INT_MEMBER, StorageType.FLOAT_MEMBER, StorageType.BOOLEAN_MEMBER, StorageType.ID_MEMBER))
                    generated_value_members.append(gen_member)
            if is_custom_member:
                result = var_type.entry_hook(result)
                if result == ContinueReadMember.result.ABORT:
                    stop_reading_members = True
            setattr(self, var_name, result)
        offset += struct.calcsize(struct_format)
        return (offset, generated_value_members, stop_reading_members)

    @classmethod
    def get_data_format(cls, game_version: GameVersion, allowed_modes: tuple[MemberAccess]=None, flatten_includes: bool=False, is_parent: bool=False):
        if False:
            while True:
                i = 10
        "\n        return all members of this exportable (a struct.)\n\n        can filter by export modes and can also return included members:\n        inherited members can either be returned as to-be-included,\n        or can be fetched and displayed as if they weren't inherited.\n        "
        for member in cls.get_data_format_members(game_version):
            if len(member) != 4:
                print(member[1])
            (export, _, _, read_type) = member
            definitively_return_member = False
            if isinstance(read_type, IncludeMembers):
                if flatten_includes:
                    yield from read_type.cls.get_data_format(game_version, allowed_modes, flatten_includes, is_parent=True)
                    continue
            elif isinstance(read_type, ContinueReadMember):
                definitively_return_member = True
            if allowed_modes:
                if export not in allowed_modes:
                    if not definitively_return_member:
                        continue
            member_entry = (is_parent,) + member
            yield member_entry

    @classmethod
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the members in this struct.\n\n        struct format specification\n        ===========================================================\n        contains a list of 4-tuples that define\n        (read_mode, var_name, storage_type, read_type)\n\n        read_mode: Tells whether to read or skip values\n        var_name: The stored name of the extracted variable.\n                  Must be unique for each ConverterObject\n        storage_type: ValueMember type for storage\n                      (see value_members.StorageType)\n        read_type: ReadMember type for reading the values from bytes\n                   (see read_members.py)\n        ===========================================================\n\n        '
        raise NotImplementedError('Subclass has not implemented this function')

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.__class__.__name__