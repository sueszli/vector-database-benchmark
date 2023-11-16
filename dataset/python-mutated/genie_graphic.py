"""
Contains structures and API-like objects for graphics from AoC.
"""
from __future__ import annotations
import typing
from ..converter_object import ConverterObject
if typing.TYPE_CHECKING:
    from openage.convert.entity_object.conversion.aoc.genie_object_container import GenieObjectContainer
    from openage.convert.value_object.read.value_members import ValueMember

class GenieGraphic(ConverterObject):
    """
    Graphic definition from a .dat file.
    """
    __slots__ = ('exists', 'subgraphics', '_refs', 'data')

    def __init__(self, graphic_id: int, full_data_set: GenieObjectContainer, members: dict[str, ValueMember]=None):
        if False:
            while True:
                i = 10
        '\n        Creates a new Genie graphic object.\n\n        :param graphic_id: The graphic id from the .dat file.\n        :type graphic_id: int\n        :param full_data_set: GenieObjectContainer instance that\n                              contains all relevant data for the conversion\n                              process.\n        :type full_data_set: class: ...dataformat.converter_object.ConverterObjectContainer\n        :param members: Members belonging to the graphic.\n        :type members: dict, optional\n        '
        super().__init__(graphic_id, members=members)
        self.data = full_data_set
        self.exists = True
        self.subgraphics: list[GenieGraphic] = []
        self._refs: list[GenieGraphic] = []

    def add_reference(self, referer: GenieGraphic) -> None:
        if False:
            while True:
                i = 10
        '\n        Add another graphic that is referencing this sprite.\n        '
        self._refs.append(referer)

    def detect_subgraphics(self) -> None:
        if False:
            print('Hello World!')
        '\n        Add references for the direct subgraphics to this object.\n        '
        graphic_deltas = self['graphic_deltas'].value
        for subgraphic in graphic_deltas:
            graphic_id = subgraphic['graphic_id'].value
            if graphic_id not in self.data.genie_graphics.keys():
                continue
            graphic = self.data.genie_graphics[graphic_id]
            self.subgraphics.append(graphic)
            graphic.add_reference(self)

    def get_animation_length(self) -> float:
        if False:
            i = 10
            return i + 15
        '\n        Returns the time taken to display all frames in this graphic.\n        '
        head_graphic = self.data.genie_graphics[self.get_id()]
        return head_graphic['frame_rate'].value * head_graphic['frame_count'].value

    def get_subgraphics(self) -> list[GenieGraphic]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the subgraphics of this graphic\n        '
        return self.subgraphics

    def get_frame_rate(self) -> float:
        if False:
            print('Hello World!')
        '\n        Returns the time taken to display a single frame in this graphic.\n        '
        head_graphic = self.data.genie_graphics[self.get_id()]
        return head_graphic['frame_rate'].value

    def is_shared(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Return True if the number of references to this graphic is >1.\n        '
        return len(self._refs) > 1

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'GenieGraphic<{self.get_id()}>'