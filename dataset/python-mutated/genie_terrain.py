"""
Contains structures and API-like objects for terrain from AoC.
"""
from __future__ import annotations
import typing
from ..converter_object import ConverterObject, ConverterObjectGroup
if typing.TYPE_CHECKING:
    from openage.convert.entity_object.conversion.aoc.genie_object_container import GenieObjectContainer
    from openage.convert.value_object.read.value_members import ValueMember

class GenieTerrainObject(ConverterObject):
    """
    Terrain definition from a .dat file.
    """
    __slots__ = ('data',)

    def __init__(self, terrain_id: int, full_data_set: GenieObjectContainer, members: dict[str, ValueMember]=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Creates a new Genie terrain object.\n\n        :param terrain_id: The index of the terrain in the .dat file's terrain\n                           block. (the index is referenced by other terrains)\n        :param full_data_set: GenieObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        :param members: An already existing member dict.\n        "
        super().__init__(terrain_id, members=members)
        self.data = full_data_set

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'GenieTerrainObject<{self.get_id()}>'

class GenieTerrainGroup(ConverterObjectGroup):
    """
    A terrain from AoE that will become an openage Terrain object.
    """
    __slots__ = ('data', 'terrain')

    def __init__(self, terrain_id: int, full_data_set: GenieObjectContainer):
        if False:
            for i in range(10):
                print('nop')
        "\n        Creates a new Genie tech group object.\n\n        :param terrain_id: The index of the terrain in the .dat file's terrain table.\n        :param full_data_set: GenieObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        "
        super().__init__(terrain_id)
        self.data = full_data_set
        self.terrain = self.data.genie_terrains[terrain_id]

    def has_subterrain(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Checks if this terrain uses a subterrain for its graphics.\n        '
        return self.terrain['terrain_replacement_id'].value > -1

    def get_subterrain(self) -> GenieTerrainObject:
        if False:
            while True:
                i = 10
        '\n        Return the subterrain used for the graphics.\n        '
        return self.data.genie_terrains[self.terrain['terrain_replacement_id'].value]

    def get_terrain(self) -> GenieTerrainObject:
        if False:
            return 10
        '\n        Return the subterrain used for the graphics.\n        '
        return self.terrain

    def __repr__(self):
        if False:
            return 10
        return f'GenieTerrainGroup<{self.get_id()}>'