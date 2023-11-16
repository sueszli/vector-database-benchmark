"""
SWGB tech objects. These extend the normal Genie techs to reflect
that SWGB techs can have unique variants for every civilization.
"""
from __future__ import annotations
import typing
from ..aoc.genie_tech import UnitUnlock, UnitLineUpgrade
if typing.TYPE_CHECKING:
    from openage.convert.entity_object.conversion.aoc.genie_object_container import GenieObjectContainer

class SWGBUnitLineUpgrade(UnitLineUpgrade):
    """
    Upgrades attributes of units/buildings or other stats in the game.
    """
    __slots__ = ('civ_unlocks',)

    def __init__(self, tech_id: int, unit_line_id: int, upgrade_target_id: int, full_data_set: GenieObjectContainer):
        if False:
            i = 10
            return i + 15
        '\n        Creates a new SWGB unit upgrade object.\n\n        :param tech_id: The internal tech_id from the .dat file.\n        :param unit_line_id: The unit line that is upgraded.\n        :param upgrade_target_id: The unit that is the result of the upgrade.\n        :param full_data_set: GenieObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        '
        super().__init__(tech_id, unit_line_id, upgrade_target_id, full_data_set)
        self.civ_unlocks: dict[int, SWGBUnitUnlock] = {}

    def add_civ_upgrade(self, other_unlock: SWGBUnitUnlock) -> None:
        if False:
            return 10
        '\n        Adds a reference to an alternative unlock tech for another civ\n        to this tech group.\n        '
        other_civ_id = other_unlock.tech['civilization_id'].value
        self.civ_unlocks[other_civ_id] = other_unlock

    def is_unique(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Techs are unique if they belong to a specific civ.\n\n        :returns: True if the civilization id is greater than zero.\n        '
        return len(self.civ_unlocks) == 0

class SWGBUnitUnlock(UnitUnlock):
    """
    Upgrades attributes of units/buildings or other stats in the game.
    """
    __slots__ = ('civ_unlocks',)

    def __init__(self, tech_id: int, line_id: int, full_data_set: GenieObjectContainer):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new SWGB unit unlock object.\n\n        :param tech_id: The internal tech_id from the .dat file.\n        :param line_id: The id of the unlocked line.\n        :param full_data_set: GenieObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        '
        super().__init__(tech_id, line_id, full_data_set)
        self.civ_unlocks: dict[int, SWGBUnitUnlock] = {}

    def add_civ_unlock(self, other_unlock: SWGBUnitUnlock) -> None:
        if False:
            print('Hello World!')
        '\n        Adds a reference to an alternative unlock tech for another civ\n        to this tech group.\n        '
        other_civ_id = other_unlock.tech['civilization_id'].value
        self.civ_unlocks[other_civ_id] = other_unlock

    def is_unique(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Techs are unique if they belong to a specific civ.\n\n        :returns: True if the civilization id is greater than zero.\n        '
        return len(self.civ_unlocks) == 0