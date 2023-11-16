from __future__ import annotations
import typing
from collections import defaultdict
from ...value_object.read.genie_structure import GenieStructure
if typing.TYPE_CHECKING:
    from openage.convert.value_object.init.game_version import GameVersion

class StringResource(GenieStructure):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.strings = defaultdict(lambda : {})

    def fill_from(self, stringtable: dict[str, dict[str, str]]) -> None:
        if False:
            return 10
        '\n        stringtable is a dict {langcode: {id: string}}\n        '
        for (lang, langstrings) in stringtable.items():
            self.strings[lang].update(langstrings)

    def get_tables(self) -> dict[str, dict[str, str]]:
        if False:
            return 10
        '\n        Returns the stringtable.\n        '
        return self.strings

    @classmethod
    def get_data_format_members(cls, game_version: GameVersion) -> tuple:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the members in this struct.\n        '
        data_format = ((True, 'id', None, 'int32_t'), (True, 'lang', None, 'char[16]'), (True, 'text', None, 'std::string'))
        return data_format