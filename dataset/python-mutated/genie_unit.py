"""
Converter objects for SWGB. Reimplements the ConverterObjectGroup
instances from AoC.
"""
from __future__ import annotations
import typing
from ..aoc.genie_unit import GenieUnitLineGroup, GenieUnitTransformGroup, GenieMonkGroup, GenieStackBuildingGroup
if typing.TYPE_CHECKING:
    from openage.convert.entity_object.conversion.aoc.genie_object_container import GenieObjectContainer

class SWGBUnitLineGroup(GenieUnitLineGroup):
    """
    A collection of GenieUnitObject types that form an "upgrade line"
    in SWGB. In comparison to AoE, there is one almost identical line
    for every civ (civ line).

    Example: Trooper Recruit->Trooper->Heavy Trooper->Repeater Trooper

    Only the civ lines will get converted to a game entity. All others
    with have their differences patched in by the civ.
    """
    __slots__ = ('civ_lines',)

    def __init__(self, line_id: int, full_data_set: GenieObjectContainer):
        if False:
            print('Hello World!')
        '\n        Creates a new SWGBUnitLineGroup.\n\n        :param line_id: Internal line obj_id in the .dat file.\n        :param full_data_set: GenieObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        '
        super().__init__(line_id, full_data_set)
        self.civ_lines: dict[int, SWGBUnitLineGroup] = {}

    def add_civ_line(self, other_line: SWGBUnitLineGroup) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Adds a reference to an alternative line from another civ\n        to this line.\n        '
        other_civ_id = other_line.get_civ_id()
        self.civ_lines[other_civ_id] = other_line

    def get_civ_id(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Returns the ID of the civ that the line belongs to.\n        '
        head_unit = self.get_head_unit()
        return head_unit['civilization_id'].value

    def is_civ_unique(self) -> bool:
        if False:
            return 10
        '\n        Groups are civ unique if there are alternative lines for this unit line..\n\n        :returns: True if alternative lines for this unit line exist.\n        '
        return len(self.civ_lines) > 0

    def is_unique(self) -> bool:
        if False:
            i = 10
            return i + 15
        "\n        Groups are unique if they belong to a specific civ.\n\n        :returns: True if the civ id is not Gaia's and no alternative lines\n                  for this unit line exist.\n        "
        return self.get_civ_id() != 0 and len(self.civ_lines) == 0 and (self.get_enabling_research_id() > -1)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'SWGBUnitLineGroup<{self.get_id()}>'

class SWGBStackBuildingGroup(GenieStackBuildingGroup):
    """
    Buildings that stack with other units and have annexes. These buildings
    are replaced by their stack unit once built.

    Examples: Gate, Command Center
    """

    def get_enabling_research_id(self) -> int:
        if False:
            print('Hello World!')
        '\n        Returns the enabling tech id of the unit\n        '
        stack_unit = self.get_stack_unit()
        stack_unit_id = stack_unit['id0'].value
        stack_unit_connection = self.data.building_connections[stack_unit_id]
        enabling_research_id = stack_unit_connection['enabling_research'].value
        return enabling_research_id

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'SWGBStackBuildingGroup<{self.get_id()}>'

class SWGBUnitTransformGroup(GenieUnitTransformGroup):
    """
    Collection of genie units that reference each other with their
    transform_id.

    Example: Cannon

    Only the civ lines will get converted to a game entity. All others
    with have their differences patched in by the civ.
    """
    __slots__ = ('civ_lines',)

    def __init__(self, line_id: int, head_unit_id: int, full_data_set: GenieObjectContainer):
        if False:
            print('Hello World!')
        '\n        Creates a new SWGB transform group.\n\n        :param head_unit_id: Internal unit obj_id of the unit that should be\n                             the initial state.\n        :param full_data_set: GenieObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        '
        super().__init__(line_id, head_unit_id, full_data_set)
        self.civ_lines: dict[int, SWGBUnitTransformGroup] = {}

    def add_civ_line(self, other_line: SWGBUnitLineGroup) -> None:
        if False:
            return 10
        '\n        Adds a reference to an alternative line from another civ\n        to this line.\n        '
        other_civ_id = other_line.get_civ_id()
        self.civ_lines[other_civ_id] = other_line

    def get_civ_id(self) -> int:
        if False:
            return 10
        '\n        Returns the ID of the civ that the line belongs to.\n        '
        head_unit = self.get_head_unit()
        return head_unit['civilization_id'].value

    def is_civ_unique(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Groups are civ unique if there are alternative lines for this unit line..\n\n        :returns: True if alternative lines for this unit line exist.\n        '
        return len(self.civ_lines) > 0

    def is_unique(self) -> bool:
        if False:
            return 10
        "\n        Groups are unique if they belong to a specific civ.\n\n        :returns: True if the civ id is not Gaia's and no alternative lines\n                  for this unit line exist.\n        "
        return False

    def get_enabling_research_id(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Returns the enabling tech id of the unit\n        '
        head_unit_connection = self.data.unit_connections[self.get_transform_unit_id()]
        enabling_research_id = head_unit_connection['enabling_research'].value
        return enabling_research_id

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'SWGBUnitTransformGroup<{self.get_id()}>'

class SWGBMonkGroup(GenieMonkGroup):
    """
    Collection of jedi/sith units and jedi/sith with holocron. The switch
    between these is hardcoded like in AoE2.

    Only the civ lines will get converted to a game entity. All others
    with have their differences patched in by the civ.
    """
    __slots__ = ('civ_lines',)

    def __init__(self, line_id: int, head_unit_id: int, switch_unit_id: int, full_data_set: GenieObjectContainer):
        if False:
            while True:
                i = 10
        '\n        Creates a new Genie monk group.\n\n        :param head_unit_id: The unit with this task will become the actual\n                             GameEntity.\n        :param switch_unit_id: This unit will be used to determine the\n                               CarryProgress objects.\n        :param full_data_set: GenieObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        '
        super().__init__(line_id, head_unit_id, switch_unit_id, full_data_set)
        self.civ_lines: dict[int, SWGBMonkGroup] = {}

    def add_civ_line(self, other_line: SWGBMonkGroup) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a reference to an alternative line from another civ\n        to this line.\n        '
        other_civ_id = other_line.get_civ_id()
        self.civ_lines[other_civ_id] = other_line

    def get_civ_id(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Returns the ID of the civ that the line belongs to.\n        '
        head_unit = self.get_head_unit()
        return head_unit['civilization_id'].value

    def is_civ_unique(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Groups are civ unique if there are alternative lines for this unit line..\n\n        :returns: True if alternative lines for this unit line exist.\n        '
        return len(self.civ_lines) > 0

    def is_unique(self) -> bool:
        if False:
            print('Hello World!')
        "\n        Groups are unique if they belong to a specific civ.\n\n        :returns: True if the civ id is not Gaia's and no alternative lines\n                  for this unit line exist.\n        "
        return False

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'SWGBMonkGroup<{self.get_id()}>'