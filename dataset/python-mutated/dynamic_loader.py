"""
Dynamically load and unload data from a file at runtime.
"""
from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from openage.convert.value_object.read.genie_structure import GenieStructure
    from openage.convert.value_object.init.game_version import GameVersion
    from openage.convert.value_object.read.value_members import ValueMember

class DynamicLoader:
    """
    Member that can be loaded and unloaded at runtime, saving
    memory in the process.
    """
    __slots__ = ('name', 'datacls', 'game_version', 'srcdata', 'offset', 'members', '_loaded')

    def __init__(self, name: str, datacls: type[GenieStructure], game_version: GameVersion, srcdata: bytes, offset: int):
        if False:
            i = 10
            return i + 15
        self.datacls = datacls
        self.game_version = game_version
        self.srcdata = srcdata
        self.offset = offset
        self._loaded = False
        self.name = name
        self.members = None

    def load(self) -> dict[str, ValueMember]:
        if False:
            print('Hello World!')
        '\n        Read the members from the provided source data.\n        '
        datacls = self.datacls()
        (_, members) = datacls.read(self.srcdata, self.offset, self.game_version, dynamic_load=False)
        self.members = {}
        for member in members:
            self.members[member.name] = member
        self._loaded = True
        return self.members

    def unload(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete the loaded members.\n        '
        del self.members
        self._loaded = False

    def __getitem__(self, key) -> ValueMember:
        if False:
            while True:
                i = 10
        '\n        Retrieve submembers from the loaded members or load them temporarily\n        if they have not been loaded previously.\n        '
        if self._loaded:
            return self.members[key]
        self.load()
        item = self.members[key]
        self.unload()
        return item

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f"DynamicLoader<{('loaded' if self._loaded else 'unloaded')}>"