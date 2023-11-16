from __future__ import annotations
import typing
from functools import cache
from ...genie_structure import GenieStructure
from ....read.member_access import READ, READ_GEN, SKIP
from ....read.read_members import SubdataMember, EnumLookupMember
from ....read.value_members import StorageType
from .lookup_dicts import EFFECT_APPLY_TYPE, CONNECTION_MODE
if typing.TYPE_CHECKING:
    from openage.convert.value_object.init.game_version import GameVersion
    from openage.convert.value_object.read.member_access import MemberAccess
    from openage.convert.value_object.read.read_members import ReadMember

class Effect(GenieStructure):

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            while True:
                i = 10
        '\n        Return the members in this struct.\n        '
        data_format = [(READ_GEN, 'type_id', StorageType.ID_MEMBER, EnumLookupMember(raw_type='int8_t', type_name='effect_apply_type', lookup_dict=EFFECT_APPLY_TYPE)), (READ_GEN, 'attr_a', StorageType.INT_MEMBER, 'int16_t'), (READ_GEN, 'attr_b', StorageType.INT_MEMBER, 'int16_t'), (READ_GEN, 'attr_c', StorageType.INT_MEMBER, 'int16_t'), (READ_GEN, 'attr_d', StorageType.FLOAT_MEMBER, 'float')]
        return data_format

class EffectBundle(GenieStructure):

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            return 10
        '\n        Return the members in this struct.\n        '
        if game_version.edition.game_id in ('AOE1DE', 'AOE2DE'):
            data_format = [(SKIP, 'name_len_debug', StorageType.INT_MEMBER, 'uint16_t'), (READ, 'name_len', StorageType.INT_MEMBER, 'uint16_t'), (SKIP, 'name', StorageType.STRING_MEMBER, 'char[name_len]')]
        else:
            data_format = [(SKIP, 'name', StorageType.STRING_MEMBER, 'char[31]')]
        data_format.extend([(READ, 'effect_count', StorageType.INT_MEMBER, 'uint16_t'), (READ_GEN, 'effects', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=Effect, length='effect_count'))])
        return data_format

class OtherConnection(GenieStructure):

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            while True:
                i = 10
        '\n        Return the members in this struct.\n        '
        data_format = [(READ_GEN, 'other_connection', StorageType.ID_MEMBER, EnumLookupMember(raw_type='int32_t', type_name='connection_mode', lookup_dict=CONNECTION_MODE))]
        return data_format

class AgeTechTree(GenieStructure):
    dynamic_load = True

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            print('Hello World!')
        '\n        Return the members in this struct.\n        '
        data_format = [(READ_GEN, 'id', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'status', StorageType.ID_MEMBER, 'int8_t')]
        if game_version.edition.game_id != 'ROR':
            data_format.extend([(READ, 'building_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'buildings', StorageType.ARRAY_ID, 'int32_t[building_count]'), (READ, 'unit_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'units', StorageType.ARRAY_ID, 'int32_t[unit_count]'), (READ, 'research_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'researches', StorageType.ARRAY_ID, 'int32_t[research_count]')])
        else:
            data_format.extend([(READ, 'building_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'buildings', StorageType.ARRAY_ID, 'int32_t[40]'), (READ, 'unit_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'units', StorageType.ARRAY_ID, 'int32_t[40]'), (READ, 'research_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'researches', StorageType.ARRAY_ID, 'int32_t[40]')])
        data_format.extend([(READ_GEN, 'connected_slots_used', StorageType.INT_MEMBER, 'int32_t')])
        if game_version.edition.game_id == 'SWGB':
            data_format.extend([(READ_GEN, 'other_connected_ids', StorageType.ARRAY_ID, 'int32_t[20]'), (READ_GEN, 'other_connections', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=OtherConnection, length=20))])
        elif game_version.edition.game_id == 'ROR':
            data_format.extend([(READ_GEN, 'other_connected_ids', StorageType.ARRAY_ID, 'int32_t[5]'), (READ_GEN, 'other_connections', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=OtherConnection, length=5))])
        else:
            data_format.extend([(READ_GEN, 'other_connected_ids', StorageType.ARRAY_ID, 'int32_t[10]'), (READ_GEN, 'other_connections', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=OtherConnection, length=10))])
        data_format.extend([(READ_GEN, 'building_level_count', StorageType.INT_MEMBER, 'int8_t')])
        if game_version.edition.game_id == 'SWGB':
            data_format.extend([(READ_GEN, 'buildings_per_zone', StorageType.ARRAY_INT, 'int8_t[20]'), (READ_GEN, 'group_length_per_zone', StorageType.ARRAY_INT, 'int8_t[20]')])
        elif game_version.edition.game_id == 'ROR':
            data_format.extend([(READ_GEN, 'buildings_per_zone', StorageType.ARRAY_INT, 'int8_t[3]'), (READ_GEN, 'group_length_per_zone', StorageType.ARRAY_INT, 'int8_t[3]')])
        else:
            data_format.extend([(READ_GEN, 'buildings_per_zone', StorageType.ARRAY_INT, 'int8_t[10]'), (READ_GEN, 'group_length_per_zone', StorageType.ARRAY_INT, 'int8_t[10]')])
        data_format.extend([(READ_GEN, 'max_age_length', StorageType.INT_MEMBER, 'int8_t'), (READ_GEN, 'line_mode', StorageType.ID_MEMBER, 'int32_t')])
        return data_format

class BuildingConnection(GenieStructure):
    dynamic_load = True

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            i = 10
            return i + 15
        '\n        Return the members in this struct.\n        '
        data_format = [(READ_GEN, 'id', StorageType.ID_MEMBER, 'int32_t'), (READ, 'status', StorageType.ID_MEMBER, 'int8_t')]
        if game_version.edition.game_id != 'ROR':
            data_format.extend([(READ, 'building_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'buildings', StorageType.ARRAY_ID, 'int32_t[building_count]'), (READ, 'unit_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'units', StorageType.ARRAY_ID, 'int32_t[unit_count]'), (READ, 'research_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'researches', StorageType.ARRAY_ID, 'int32_t[research_count]')])
        else:
            data_format.extend([(READ, 'building_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'buildings', StorageType.ARRAY_ID, 'int32_t[40]'), (READ, 'unit_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'units', StorageType.ARRAY_ID, 'int32_t[40]'), (READ, 'research_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'researches', StorageType.ARRAY_ID, 'int32_t[40]')])
        data_format.extend([(READ_GEN, 'connected_slots_used', StorageType.INT_MEMBER, 'int32_t')])
        if game_version.edition.game_id == 'SWGB':
            data_format.extend([(READ_GEN, 'other_connected_ids', StorageType.ARRAY_ID, 'int32_t[20]'), (READ_GEN, 'other_connections', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=OtherConnection, length=20))])
        elif game_version.edition.game_id == 'ROR':
            data_format.extend([(READ_GEN, 'other_connected_ids', StorageType.ARRAY_ID, 'int32_t[5]'), (READ_GEN, 'other_connections', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=OtherConnection, length=5))])
        else:
            data_format.extend([(READ_GEN, 'other_connected_ids', StorageType.ARRAY_ID, 'int32_t[10]'), (READ_GEN, 'other_connections', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=OtherConnection, length=10))])
        data_format.extend([(READ_GEN, 'location_in_age', StorageType.ID_MEMBER, 'int8_t'), (READ_GEN, 'unit_techs_total', StorageType.ARRAY_INT, 'int8_t[5]'), (READ_GEN, 'unit_techs_first', StorageType.ARRAY_INT, 'int8_t[5]'), (READ_GEN, 'line_mode', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'enabling_research', StorageType.ID_MEMBER, 'int32_t')])
        return data_format

class UnitConnection(GenieStructure):
    dynamic_load = True

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            print('Hello World!')
        '\n        Return the members in this struct.\n        '
        data_format = [(READ_GEN, 'id', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'status', StorageType.ID_MEMBER, 'int8_t'), (READ_GEN, 'upper_building', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'connected_slots_used', StorageType.INT_MEMBER, 'int32_t')]
        if game_version.edition.game_id == 'SWGB':
            data_format.extend([(READ_GEN, 'other_connected_ids', StorageType.ARRAY_ID, 'int32_t[20]'), (READ_GEN, 'other_connections', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=OtherConnection, length=20))])
        elif game_version.edition.game_id == 'ROR':
            data_format.extend([(READ_GEN, 'other_connected_ids', StorageType.ARRAY_ID, 'int32_t[5]'), (READ_GEN, 'other_connections', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=OtherConnection, length=5))])
        else:
            data_format.extend([(READ_GEN, 'other_connected_ids', StorageType.ARRAY_ID, 'int32_t[10]'), (READ_GEN, 'other_connections', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=OtherConnection, length=10))])
        data_format.extend([(READ_GEN, 'vertical_line', StorageType.ID_MEMBER, 'int32_t')])
        if game_version.edition.game_id != 'ROR':
            data_format.extend([(READ, 'unit_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'units', StorageType.ARRAY_ID, 'int32_t[unit_count]')])
        else:
            data_format.extend([(READ, 'unit_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'units', StorageType.ARRAY_ID, 'int32_t[40]')])
        data_format.extend([(READ_GEN, 'location_in_age', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'required_research', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'line_mode', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'enabling_research', StorageType.ID_MEMBER, 'int32_t')])
        return data_format

class ResearchConnection(GenieStructure):
    dynamic_load = True

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            return 10
        '\n        Return the members in this struct.\n        '
        data_format = [(READ_GEN, 'id', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'status', StorageType.ID_MEMBER, 'int8_t'), (READ_GEN, 'upper_building', StorageType.ID_MEMBER, 'int32_t')]
        if game_version.edition.game_id != 'ROR':
            data_format.extend([(READ, 'building_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'buildings', StorageType.ARRAY_ID, 'int32_t[building_count]'), (READ, 'unit_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'units', StorageType.ARRAY_ID, 'int32_t[unit_count]'), (READ, 'research_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'researches', StorageType.ARRAY_ID, 'int32_t[research_count]')])
        else:
            data_format.extend([(READ, 'building_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'buildings', StorageType.ARRAY_ID, 'int32_t[40]'), (READ, 'unit_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'units', StorageType.ARRAY_ID, 'int32_t[40]'), (READ, 'research_count', StorageType.INT_MEMBER, 'uint8_t'), (READ_GEN, 'researches', StorageType.ARRAY_ID, 'int32_t[40]')])
        data_format.extend([(READ_GEN, 'connected_slots_used', StorageType.INT_MEMBER, 'int32_t')])
        if game_version.edition.game_id == 'SWGB':
            data_format.extend([(READ_GEN, 'other_connected_ids', StorageType.ARRAY_ID, 'int32_t[20]'), (READ_GEN, 'other_connections', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=OtherConnection, length=20))])
        elif game_version.edition.game_id == 'ROR':
            data_format.extend([(READ_GEN, 'other_connected_ids', StorageType.ARRAY_ID, 'int32_t[5]'), (READ_GEN, 'other_connections', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=OtherConnection, length=5))])
        else:
            data_format.extend([(READ_GEN, 'other_connected_ids', StorageType.ARRAY_ID, 'int32_t[10]'), (READ_GEN, 'other_connections', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=OtherConnection, length=10))])
        data_format.extend([(READ_GEN, 'vertical_line', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'location_in_age', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'line_mode', StorageType.ID_MEMBER, 'int32_t')])
        return data_format