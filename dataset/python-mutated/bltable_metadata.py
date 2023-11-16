"""
Blendtable definition file.
"""
from __future__ import annotations
import typing
from ..data_definition import DataDefinition
FORMAT_VERSION = '1'

class BlendtableMetadata(DataDefinition):
    """
    Collects blentable metadata and can format it
    as a .bltable custom format
    """

    def __init__(self, targetdir: str, filename: str):
        if False:
            print('Hello World!')
        super().__init__(targetdir, filename)
        self.blendtable: tuple = None
        self.patterns: dict[int, dict[str, typing.Any]] = {}

    def add_pattern(self, pattern_id: int, filename: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Define a pattern in the table.\n\n        :param pattern_id: Pattern identifier.\n        :type pattern_id: int\n        :param filename: Path to the pattern file.\n        :type filename: str\n        '
        self.patterns[pattern_id] = {'pattern_id': pattern_id, 'filename': filename}

    def set_blendtabe(self, table: tuple) -> None:
        if False:
            return 10
        '\n        Set the blendtable. This expects a tuple of integers with nxn entries.\n\n        :param table: Blending lookup table.\n        :type table: tuple\n        '
        self.blendtable = table
        self._check_table()

    def dump(self) -> str:
        if False:
            return 10
        output_str = ''
        output_str += '# openage blendtable definition file\n\n'
        output_str += f'version {FORMAT_VERSION}\n\n'
        output_str += 'blendtable [\n'
        table_width = self._get_table_width()
        for idx in range(table_width):
            row_entries = self.blendtable[idx * table_width:(idx + 1) * table_width]
            output_str += f"{' '.join(row_entries)}\n"
        output_str += ']\n\n'
        for pattern in self.patterns.values():
            output_str += f"pattern {pattern['pattern_id']} {pattern['filename']}\n"
        output_str += '\n'
        return output_str

    def _get_table_width(self) -> int:
        if False:
            print('Hello World!')
        '\n        Get the width of the blending table.\n        '
        table_size = len(self.blendtable)
        left = table_size
        right = (left + 1) // 2
        while right < left:
            left = right
            right = (left + table_size // left) // 2
        return left

    def _check_table(self) -> typing.Union[None, typing.NoReturn]:
        if False:
            i = 10
            return i + 15
        '\n        Check if the blending table is a nxn matrix.\n        '
        table_width = self._get_table_width()
        if table_width * table_width != len(self.blendtable):
            raise ValueError(f'blendtable entries malformed: {len(self.blendtable)} is not an integer square')

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'BlendtableMetadata<{self.filename}>'