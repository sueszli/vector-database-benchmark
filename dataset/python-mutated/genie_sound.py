"""
Contains structures and API-like objects for sounds from AoC.
"""
from __future__ import annotations
import typing
from ..converter_object import ConverterObject
if typing.TYPE_CHECKING:
    from openage.convert.entity_object.conversion.aoc.genie_object_container import GenieObjectContainer
    from openage.convert.value_object.read.value_members import ValueMember

class GenieSound(ConverterObject):
    """
    Sound definition from a .dat file.
    """
    __slots__ = ('data',)

    def __init__(self, sound_id: int, full_data_set: GenieObjectContainer, members: dict[str, ValueMember]=None):
        if False:
            print('Hello World!')
        '\n        Creates a new Genie sound object.\n\n        :param sound_id: The sound id from the .dat file.\n        :param full_data_set: GenieObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        :param members: An already existing member dict.\n        '
        super().__init__(sound_id, members=members)
        self.data = full_data_set

    def get_sounds(self, civ_id: int=-1) -> list[int]:
        if False:
            while True:
                i = 10
        '\n        Return sound resource ids for the associated DRS file.\n\n        :param civ_id: If specified, only return sounds that belong to this civ.\n        :type civ_id: int\n        '
        sound_ids = []
        sound_items = self['sound_items'].value
        for item in sound_items:
            item_civ_id = item['civilization_id'].value
            if not item_civ_id == civ_id:
                continue
            sound_id = item['resource_id'].value
            sound_ids.append(sound_id)
        return sound_ids

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'GenieSound<{self.get_id()}>'