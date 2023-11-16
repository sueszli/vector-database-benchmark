"""
References a graphic in the game that has to be converted.
"""
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from openage.convert.entity_object.conversion.converter_object import ConverterObjectContainer
    from openage.convert.entity_object.conversion.converter_object import ConverterObject
    from openage.convert.entity_object.conversion.aoc.genie_terrain import GenieTerrainObject

class CombinedTerrain:
    """
    Collection of terrain information for openage files.

    This will become a spritesheet texture with a terrain file.
    """
    __slots__ = ('terrain_id', 'filename', 'data', 'metadata', '_refs')

    def __init__(self, terrain_id: int, filename: str, full_data_set: ConverterObjectContainer):
        if False:
            print('Hello World!')
        '\n        Creates a new CombinedTerrain instance.\n\n        :param terrain_id: The index of the terrain that references the sprite.\n        :type terrain_id: int\n        :param filename: Name of the terrain and definition file.\n        :type filename: str\n        :param full_data_set: GenieObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        :type full_data_set: class: ...dataformat.converter_object.ConverterObjectContainer\n        '
        self.terrain_id = terrain_id
        self.filename = filename
        self.data = full_data_set
        self.metadata = None
        self._refs = []

    def add_reference(self, referer: ConverterObject) -> None:
        if False:
            while True:
                i = 10
        '\n        Add an object that is referencing this terrain.\n        '
        self._refs.append(referer)

    def get_filename(self) -> str:
        if False:
            print('Hello World!')
        '\n        Returns the destination filename of the terrain.\n        '
        return self.filename

    def get_terrain(self) -> GenieTerrainObject:
        if False:
            return 10
        '\n        Returns the terrain referenced by this terrain sprite.\n        '
        return self.data.genie_terrains[self.terrain_id]

    def get_id(self) -> int:
        if False:
            return 10
        '\n        Returns the terrain id of the terrain.\n        '
        return self.terrain_id

    def get_relative_terrain_location(self) -> str:
        if False:
            return 10
        '\n        Return the terrain file location relative to where the file\n        is expected to be in the modpack.\n        '
        if len(self._refs) >= 1:
            return f'./graphics/{self.filename}.terrain'
        return None

    def remove_reference(self, referer: ConverterObject) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove an object that is referencing this sprite.\n        '
        self._refs.remove(referer)

    def resolve_graphics_location(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns the planned location in the modpack of the image file\n        referenced by the terrain file.\n        '
        return self.resolve_terrain_location()

    def resolve_terrain_location(self) -> str:
        if False:
            print('Hello World!')
        '\n        Returns the planned location of the definition file in the modpack.\n        '
        if len(self._refs) >= 1:
            return f"{self._refs[0].get_file_location()[0]}{'graphics/'}"
        return None

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'CombinedTerrain<{self.terrain_id}>'