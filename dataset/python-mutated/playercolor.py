from __future__ import annotations
import typing
from functools import cache
from ...genie_structure import GenieStructure
from ....read.member_access import READ_GEN, SKIP
from ....read.value_members import StorageType
if typing.TYPE_CHECKING:
    from openage.convert.value_object.init.game_version import GameVersion
    from openage.convert.value_object.read.member_access import MemberAccess
    from openage.convert.value_object.read.read_members import ReadMember

class PlayerColor(GenieStructure):
    dynamic_load = True

    @classmethod
    @cache
    def get_data_format_members(cls, game_version: GameVersion) -> list[tuple[MemberAccess, str, StorageType, typing.Union[str, ReadMember]]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the members in this struct.\n        '
        if game_version.edition.game_id not in ('ROR', 'AOE1DE'):
            data_format = [(READ_GEN, 'id', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'player_color_base', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'outline_color', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'unit_selection_color1', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'unit_selection_color2', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'minimap_color1', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'minimap_color2', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'minimap_color3', StorageType.ID_MEMBER, 'int32_t'), (READ_GEN, 'statistics_text_color', StorageType.ID_MEMBER, 'int32_t')]
        else:
            data_format = [(SKIP, 'name', StorageType.STRING_MEMBER, 'char[30]'), (READ_GEN, 'id', StorageType.ID_MEMBER, 'int16_t'), (READ_GEN, 'resource_id', StorageType.ID_MEMBER, 'int16_t'), (READ_GEN, 'minimap_color', StorageType.ID_MEMBER, 'uint8_t'), (READ_GEN, 'type', StorageType.ID_MEMBER, 'uint8_t')]
        return data_format