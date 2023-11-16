"""
Output format specification for data to write.
"""
from __future__ import annotations
import typing

class DataDefinition:
    """
    Contains a data definition that can then be
    formatted to an arbitrary output file.
    """

    def __init__(self, targetdir: str, filename: str):
        if False:
            return 10
        '\n        Creates a new data definition.\n\n        :param targetdir: Relative path to the export directory.\n        :type targetdir: str\n        :param filename: Filename of the resulting file.\n        :type filename: str\n        '
        self.targetdir = targetdir
        self.filename = filename

    def dump(self) -> typing.NoReturn:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a human-readable string that can be written to a file.\n        '
        raise NotImplementedError(f'{type(self)} has not implemented dump() method')

    def set_filename(self, filename: str) -> None:
        if False:
            return 10
        '\n        Sets the filename for the file.\n\n        :param filename: Filename of the resuilting file.\n        :type filename: str\n        '
        if not isinstance(filename, str):
            raise ValueError(f'str expected as filename, not {type(filename)}')
        self.filename = filename

    def set_targetdir(self, targetdir: str) -> None:
        if False:
            print('Hello World!')
        '\n        Sets the target directory for the file.\n\n        :param targetdir: Relative path to the export directory.\n        :type targetdir: str\n        '
        if not isinstance(targetdir, str):
            raise ValueError('str expected as targetdir')
        self.targetdir = targetdir

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'DataDefinition<{type(self)}>'