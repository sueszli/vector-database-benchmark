"""
References a graphic in the game that has to be converted.
"""
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from openage.convert.entity_object.conversion.converter_object import ConverterObjectContainer
    from openage.convert.entity_object.conversion.converter_object import ConverterObject
    from openage.convert.entity_object.conversion.aoc.genie_graphic import GenieGraphic

class CombinedSprite:
    """
    Collection of sprite information for openage files.

    This will become a spritesheet texture with a sprite file.
    """
    __slots__ = ('head_sprite_id', 'filename', 'data', 'metadata', '_refs')

    def __init__(self, head_sprite_id: int, filename: str, full_data_set: ConverterObjectContainer):
        if False:
            i = 10
            return i + 15
        '\n        Creates a new CombinedSprite instance.\n\n        :param head_sprite_id: The id of the top level graphic of this sprite.\n        :type head_sprite_id: int\n        :param filename: Name of the sprite and definition file.\n        :type filename: str\n        :param full_data_set: ConverterObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        :type full_data_set: class: ...dataformat.converter_object.ConverterObjectContainer\n        '
        self.head_sprite_id = head_sprite_id
        self.filename = filename
        self.data = full_data_set
        self.metadata = None
        self._refs = []

    def add_reference(self, referer: ConverterObject) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Add an object that is referencing this sprite.\n        '
        self._refs.append(referer)

    def get_filename(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Returns the desired filename of the sprite.\n        '
        return self.filename

    def get_graphics(self) -> list[GenieGraphic]:
        if False:
            while True:
                i = 10
        '\n        Return all graphics referenced by this sprite.\n        '
        graphics = [self.data.genie_graphics[self.head_sprite_id]]
        graphics.extend(self.data.genie_graphics[self.head_sprite_id].get_subgraphics())
        existing_graphics = []
        for graphic in graphics:
            if graphic.exists:
                existing_graphics.append(graphic)
        return existing_graphics

    def get_id(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Returns the head sprite ID of the sprite.\n        '
        return self.head_sprite_id

    def get_relative_sprite_location(self) -> str:
        if False:
            print('Hello World!')
        '\n        Return the sprite file location relative to where the file\n        is expected to be in the modpack.\n        '
        if len(self._refs) > 1:
            return f'../shared/graphics/{self.filename}.sprite'
        if len(self._refs) == 1:
            return f'./graphics/{self.filename}.sprite'
        return None

    def remove_reference(self, referer: ConverterObject) -> None:
        if False:
            print('Hello World!')
        '\n        Remove an object that is referencing this sprite.\n        '
        self._refs.remove(referer)

    def resolve_graphics_location(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Returns the planned location in the modpack of all image files\n        referenced by the sprite.\n        '
        location_dict = {}
        for graphic in self.get_graphics():
            if graphic.is_shared():
                location_dict.update({graphic.get_id(): 'data/game_entity/shared/graphics/'})
            else:
                location_dict.update({graphic.get_id(): self.resolve_sprite_location()})
        return location_dict

    def resolve_sprite_location(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the planned location of the definition file in the modpack.\n        '
        if len(self._refs) > 1:
            return 'data/game_entity/shared/graphics/'
        if len(self._refs) == 1:
            return f"{self._refs[0].get_file_location()[0]}{'graphics/'}"
        return None

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'CombinedSprite<{self.head_sprite_id}>'