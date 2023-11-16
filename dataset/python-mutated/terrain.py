from __future__ import annotations
import typing
from functools import cache
from ...genie_structure import GenieStructure
from ....read.member_access import READ, READ_GEN, SKIP
from ....read.read_members import ArrayMember, SubdataMember, IncludeMembers
from ....read.value_members import StorageType
if typing.TYPE_CHECKING:
    from openage.convert.value_object.init.game_version import GameVersion
    from openage.convert.value_object.read.member_access import MemberAccess
    from openage.convert.value_object.read.read_members import ReadMember

class FrameData(GenieStructure):

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            return 10
        '\n        Return the members in this struct.\n        '
        data_format = [(READ_GEN, 'frame_count', StorageType.INT_MEMBER, 'int16_t'), (READ_GEN, 'angle_count', StorageType.INT_MEMBER, 'int16_t'), (READ_GEN, 'shape_id', StorageType.ID_MEMBER, 'int16_t')]
        return data_format

class TerrainPassGraphic(GenieStructure):

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            return 10
        '\n        Return the members in this struct.\n        '
        data_format = [(READ_GEN, 'slp_id_exit_tile', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'slp_id_enter_tile', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'slp_id_walk_tile', StorageType.ID_MEMBER, 'int32_t')]
        if game_version.edition.game_id == 'SWGB':
            data_format.append((READ_GEN, 'walk_sprite_rate', StorageType.FLOAT_MEMBER, 'float'))
        else:
            data_format.append((READ_GEN, 'replication_amount', StorageType.INT_MEMBER, 'int32_t'))
        return data_format

class TerrainRestriction(GenieStructure):
    """
    access policies for units on specific terrain.
    """

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            return 10
        '\n        Return the members in this struct.\n        '
        data_format = [(READ_GEN, 'accessible_dmgmultiplier', StorageType.ARRAY_FLOAT, 'float[terrain_count]')]
        if game_version.edition.game_id != 'ROR':
            data_format.append((READ_GEN, 'pass_graphics', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=TerrainPassGraphic, length='terrain_count')))
        return data_format

class TerrainAnimation(GenieStructure):

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            return 10
        '\n        Return the members in this struct.\n        '
        data_format = [(READ_GEN, 'is_animated', StorageType.BOOLEAN_MEMBER, 'int8_t'), (READ_GEN, 'animation_frame_count', StorageType.INT_MEMBER, 'int16_t'), (READ_GEN, 'pause_frame_count', StorageType.INT_MEMBER, 'int16_t'), (READ_GEN, 'interval', StorageType.FLOAT_MEMBER, 'float'), (READ_GEN, 'pause_between_loops', StorageType.FLOAT_MEMBER, 'float'), (READ_GEN, 'frame', StorageType.INT_MEMBER, 'int16_t'), (READ_GEN, 'draw_frame', StorageType.INT_MEMBER, 'int16_t'), (READ_GEN, 'animate_last', StorageType.FLOAT_MEMBER, 'float'), (READ_GEN, 'frame_changed', StorageType.BOOLEAN_MEMBER, 'int8_t'), (READ_GEN, 'drawn', StorageType.BOOLEAN_MEMBER, 'int8_t')]
        return data_format

class Terrain(GenieStructure):
    dynamic_load = True

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            while True:
                i = 10
        '\n        Return the members in this struct.\n        '
        data_format = [(READ_GEN, 'enabled', StorageType.BOOLEAN_MEMBER, 'int8_t'), (READ_GEN, 'random', StorageType.INT_MEMBER, 'int8_t')]
        if game_version.edition.game_id in ('AOE1DE', 'AOE2DE'):
            data_format.extend([(READ_GEN, 'is_water', StorageType.BOOLEAN_MEMBER, 'int8_t'), (READ_GEN, 'hide_in_editor', StorageType.BOOLEAN_MEMBER, 'int8_t'), (READ_GEN, 'string_id', StorageType.ID_MEMBER, 'int32_t')])
            if game_version.edition.game_id == 'AOE1DE':
                data_format.extend([(READ_GEN, 'blend_priority', StorageType.ID_MEMBER, 'int16_t'), (READ_GEN, 'blend_type', StorageType.ID_MEMBER, 'int16_t')])
            data_format.extend([(SKIP, 'internal_name_len_debug', StorageType.INT_MEMBER, 'uint16_t'), (READ, 'internal_name_len', StorageType.INT_MEMBER, 'uint16_t'), (READ_GEN, 'internal_name', StorageType.STRING_MEMBER, 'char[internal_name_len]'), (SKIP, 'filename_len_debug', StorageType.INT_MEMBER, 'uint16_t'), (READ, 'filename_len', StorageType.INT_MEMBER, 'uint16_t'), (READ_GEN, 'filename', StorageType.STRING_MEMBER, 'char[filename_len]')])
        elif game_version.edition.game_id == 'SWGB':
            data_format.extend([(READ_GEN, 'internal_name', StorageType.STRING_MEMBER, 'char[17]'), (READ_GEN, 'filename', StorageType.STRING_MEMBER, 'char[17]')])
        else:
            data_format.extend([(READ_GEN, 'internal_name', StorageType.STRING_MEMBER, 'char[13]'), (READ_GEN, 'filename', StorageType.STRING_MEMBER, 'char[13]')])
        data_format.extend([(READ_GEN, 'slp_id', StorageType.ID_MEMBER, 'int32_t'), (SKIP, 'shape_ptr', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'sound_id', StorageType.ID_MEMBER, 'int32_t')])
        if game_version.edition.game_id == 'AOE2DE':
            data_format.extend([(READ_GEN, 'wwise_sound_id', StorageType.ID_MEMBER, 'uint32_t'), (READ_GEN, 'wwise_stop_sound_id', StorageType.ID_MEMBER, 'uint32_t')])
        if game_version.edition.game_id not in ('ROR', 'AOE1DE'):
            data_format.extend([(READ_GEN, 'blend_priority', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'blend_mode', StorageType.ID_MEMBER, 'int32_t')])
            if game_version.edition.game_id == 'AOE2DE':
                data_format.extend([(SKIP, 'overlay_mask_name_len_debug', StorageType.INT_MEMBER, 'uint16_t'), (READ, 'overlay_mask_name_len', StorageType.INT_MEMBER, 'uint16_t'), (READ_GEN, 'overlay_mask_name', StorageType.STRING_MEMBER, 'char[overlay_mask_name_len]')])
        data_format.extend([(READ_GEN, 'map_color_hi', StorageType.ID_MEMBER, 'uint8_t'), (READ_GEN, 'map_color_med', StorageType.ID_MEMBER, 'uint8_t'), (READ_GEN, 'map_color_low', StorageType.ID_MEMBER, 'uint8_t'), (READ_GEN, 'map_color_cliff_lt', StorageType.ID_MEMBER, 'uint8_t'), (READ_GEN, 'map_color_cliff_rt', StorageType.ID_MEMBER, 'uint8_t'), (READ_GEN, 'passable_terrain', StorageType.ID_MEMBER, 'int8_t'), (READ_GEN, 'impassable_terrain', StorageType.ID_MEMBER, 'int8_t'), (READ_GEN, None, None, IncludeMembers(cls=TerrainAnimation)), (READ_GEN, 'elevation_graphics', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=FrameData, length=19)), (READ_GEN, 'terrain_replacement_id', StorageType.ID_MEMBER, 'int16_t'), (READ_GEN, 'terrain_to_draw0', StorageType.ID_MEMBER, 'int16_t'), (READ_GEN, 'terrain_to_draw1', StorageType.ID_MEMBER, 'int16_t')])
        if game_version.edition.game_id == 'AOE2DE':
            data_format.append((READ_GEN, 'terrain_unit_masked_density', StorageType.ARRAY_INT, 'int16_t[30]'))
        elif game_version.edition.game_id == 'SWGB':
            data_format.append((READ_GEN, 'borders', StorageType.ARRAY_INT, ArrayMember('int16_t', 55)))
        elif game_version.edition.game_id == 'AOE1DE':
            data_format.append((READ_GEN, 'borders', StorageType.ARRAY_INT, ArrayMember('int16_t', 96)))
        elif game_version.edition.game_id == 'HDEDITION':
            if len(game_version.expansions) > 0:
                data_format.append((READ_GEN, 'borders', StorageType.ARRAY_INT, ArrayMember('int16_t', 100)))
            else:
                data_format.append((READ_GEN, 'borders', StorageType.ARRAY_INT, ArrayMember('int16_t', 42)))
        elif game_version.edition.game_id == 'AOC':
            data_format.append((READ_GEN, 'borders', StorageType.ARRAY_INT, ArrayMember('int16_t', 42)))
        else:
            data_format.append((READ_GEN, 'borders', StorageType.ARRAY_INT, ArrayMember('int16_t', 32)))
        data_format.extend([(READ_GEN, 'terrain_unit_id', StorageType.ARRAY_ID, 'int16_t[30]'), (READ_GEN, 'terrain_unit_density', StorageType.ARRAY_INT, 'int16_t[30]'), (READ_GEN, 'terrain_placement_flag', StorageType.ARRAY_BOOL, 'int8_t[30]'), (READ_GEN, 'terrain_units_used_count', StorageType.INT_MEMBER, 'int16_t')])
        if game_version.edition.game_id != 'SWGB':
            data_format.append((READ, 'phantom', StorageType.INT_MEMBER, 'int16_t'))
        return data_format

class TerrainBorder(GenieStructure):
    dynamic_load = True

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            print('Hello World!')
        '\n        Return the members in this struct.\n        '
        data_format = [(READ_GEN, 'enabled', StorageType.BOOLEAN_MEMBER, 'int8_t'), (READ_GEN, 'random', StorageType.INT_MEMBER, 'int8_t'), (READ_GEN, 'internal_name', StorageType.STRING_MEMBER, 'char[13]'), (READ_GEN, 'filename', StorageType.STRING_MEMBER, 'char[13]'), (READ_GEN, 'slp_id', StorageType.ID_MEMBER, 'int32_t'), (SKIP, 'shape_ptr', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'sound_id', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'color', StorageType.ARRAY_ID, 'uint8_t[3]'), (READ_GEN, None, None, IncludeMembers(cls=TerrainAnimation)), (READ_GEN, 'frames', StorageType.ARRAY_CONTAINER, SubdataMember(ref_type=FrameData, length=19 * 12)), (SKIP, 'draw_tile', StorageType.INT_MEMBER, 'int16_t'), (READ_GEN, 'underlay_terrain', StorageType.ID_MEMBER, 'int16_t'), (READ_GEN, 'border_style', StorageType.INT_MEMBER, 'int16_t')]
        return data_format

class TileSize(GenieStructure):

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            return 10
        '\n        Return the members in this struct.\n        '
        data_format = [(READ_GEN, 'width', StorageType.INT_MEMBER, 'int16_t'), (READ_GEN, 'height', StorageType.INT_MEMBER, 'int16_t'), (READ_GEN, 'delta_z', StorageType.INT_MEMBER, 'int16_t')]
        return data_format