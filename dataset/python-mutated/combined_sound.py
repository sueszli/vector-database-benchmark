"""
References a sound in the game that has to be converted.
"""
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from openage.convert.entity_object.conversion.converter_object import ConverterObjectContainer
    from openage.convert.entity_object.conversion.converter_object import ConverterObject

class CombinedSound:
    """
    Collection of sound information for openage files.
    """
    __slots__ = ('head_sound_id', 'file_id', 'filename', 'data', 'genie_sound', '_refs')

    def __init__(self, head_sound_id: int, file_id: int, filename: str, full_data_set: ConverterObjectContainer):
        if False:
            print('Hello World!')
        '\n        Creates a new CombinedSound instance.\n\n        :param head_sound_id: The id of the GenieSound object of this sound.\n        :type head_sound_id: int\n        :param file_id: The id of the file resource in the GenieSound.\n        :type file_id: int\n        :param filename: Name of the sound file.\n        :type filename: str\n        :param full_data_set: ConverterObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        :type full_data_set: class: ...dataformat.converter_object.ConverterObjectContainer\n        '
        self.head_sound_id = head_sound_id
        self.file_id = file_id
        self.filename = filename
        self.data = full_data_set
        self.genie_sound = self.data.genie_sounds[self.head_sound_id]
        self._refs = []

    def add_reference(self, referer: ConverterObject) -> None:
        if False:
            return 10
        '\n        Add an object that is referencing this sound.\n        '
        self._refs.append(referer)

    def get_filename(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Returns the desired filename of the sprite.\n        '
        return self.filename

    def get_file_id(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Returns the ID of the sound file in the game folder.\n        '
        return self.file_id

    def get_id(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Returns the ID of the sound object in the .dat.\n        '
        return self.head_sound_id

    def get_relative_file_location(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Return the sound file location relative to where the file\n        is expected to be in the modpack.\n        '
        if len(self._refs) > 1:
            return f'../shared/sounds/{self.filename}.opus'
        if len(self._refs) == 1:
            return f'./sounds/{self.filename}.opus'
        return None

    def resolve_sound_location(self) -> str:
        if False:
            return 10
        '\n        Returns the planned location of the sound file in the modpack.\n        '
        if len(self._refs) > 1:
            return 'data/game_entity/shared/sounds/'
        if len(self._refs) == 1:
            return f"{self._refs[0].get_file_location()[0]}{'sounds/'}"
        return None

    def remove_reference(self, referer: ConverterObject) -> None:
        if False:
            print('Hello World!')
        '\n        Remove an object that is referencing this sound.\n        '
        self._refs.remove(referer)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'CombinedSound<{self.head_sound_id}>'