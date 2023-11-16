from os.path import relpath
from coala_utils.decorators import enforce_signature, get_public_members
from coalib.results.SourcePosition import SourcePosition
from coalib.results.TextRange import TextRange
from coalib.results.AbsolutePosition import AbsolutePosition

class SourceRange(TextRange):

    @enforce_signature
    def __init__(self, start: SourcePosition, end: (SourcePosition, None)=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new SourceRange.\n\n        :param start:       A SourcePosition indicating the start of the range.\n        :param end:         A SourcePosition indicating the end of the range.\n                            If ``None`` is given, the start object will be used\n                            here. end must be in the same file and be greater\n                            than start as negative ranges are not allowed.\n        :raises TypeError:  Raised when\n                            - start is not of type SourcePosition.\n                            - end is neither of type SourcePosition, nor is it\n                              None.\n        :raises ValueError: Raised when file of start and end mismatch.\n        '
        TextRange.__init__(self, start, end)
        if self.start.file != self.end.file:
            raise ValueError('File of start and end position do not match.')

    @classmethod
    def from_values(cls, file, start_line=None, start_column=None, end_line=None, end_column=None):
        if False:
            while True:
                i = 10
        start = SourcePosition(file, start_line, start_column)
        if end_line or (end_column and end_column > start_column):
            end = SourcePosition(file, end_line if end_line else start_line, end_column)
        else:
            end = None
        return cls(start, end)

    @classmethod
    @enforce_signature
    def from_absolute_position(cls, file: str, position_start: AbsolutePosition, position_end: (AbsolutePosition, None)=None):
        if False:
            return 10
        '\n        Creates a SourceRange from a start and end positions.\n\n        :param file:           Name of the file.\n        :param position_start: Start of range given by AbsolutePosition.\n        :param position_end:   End of range given by AbsolutePosition or None.\n        '
        start = SourcePosition(file, position_start.line, position_start.column)
        end = None
        if position_end:
            end = SourcePosition(file, position_end.line, position_end.column)
        return cls(start, end)

    @property
    def file(self):
        if False:
            for i in range(10):
                print('nop')
        return self.start.file

    @enforce_signature
    def renamed_file(self, file_diff_dict: dict):
        if False:
            print('Hello World!')
        '\n        Retrieves the filename this source range refers to while taking the\n        possible file renamings in the given file_diff_dict into account:\n\n        :param file_diff_dict: A dictionary with filenames as key and their\n                               associated Diff objects as values.\n        '
        diff = file_diff_dict.get(self.file)
        if diff is None:
            return self.file
        return diff.rename if diff.rename is not False else self.file

    def expand(self, file_contents):
        if False:
            i = 10
            return i + 15
        '\n        Passes a new SourceRange that covers the same area of a file as this\n        one would. All values of None get replaced with absolute values.\n\n        values of None will be interpreted as follows:\n        self.start.line is None:   -> 1\n        self.start.column is None: -> 1\n        self.end.line is None:     -> last line of file\n        self.end.column is None:   -> last column of self.end.line\n\n        :param file_contents: File contents of the applicable file\n        :return:              TextRange with absolute values\n        '
        tr = TextRange.expand(self, file_contents)
        return SourceRange.from_values(self.file, tr.start.line, tr.start.column, tr.end.line, tr.end.column)

    @enforce_signature
    def affected_source(self, file_dict: dict):
        if False:
            print('Hello World!')
        "\n        Tells which lines are affected in a specified file within a given range.\n\n        >>> from os.path import abspath\n        >>> sr = SourceRange.from_values('file_name', start_line=2, end_line=2)\n        >>> sr.affected_source({\n        ...     abspath('file_name'): ('def fun():\\n', '    x = 2  \\n')\n        ... })\n        ('    x = 2  \\n',)\n\n        If more than one line is affected.\n\n        >>> sr = SourceRange.from_values('file_name', start_line=2, end_line=3)\n        >>> sr.affected_source({\n        ...     abspath('file_name'): ('def fun():\\n',\n        ...                            '    x = 2  \\n', '    print(x)  \\n')\n        ... })\n        ('    x = 2  \\n', '    print(x)  \\n')\n\n        If the file indicated at the source range is not in the `file_dict` or\n        the lines are not given, this will return `None`:\n\n        >>> sr = SourceRange.from_values('file_name_not_present',\n        ...     start_line=2, end_line=2)\n        >>> sr.affected_source({abspath('file_name'):\n        ...     ('def fun():\\n', '    x = 2  \\n')})\n\n        :param file_dict:\n            It is a dictionary where the file names are the keys and\n            the contents of the files are the values(which is of type tuple).\n        :return:\n            A tuple of affected lines in the specified file.\n            If the file is not affected or the file is not present in\n            ``file_dict`` return ``None``.\n        "
        if self.start.file in file_dict and self.start.line and self.end.line:
            return file_dict[self.start.file][self.start.line - 1:self.end.line]

    def __json__(self, use_relpath=False):
        if False:
            while True:
                i = 10
        _dict = get_public_members(self)
        if use_relpath:
            _dict['file'] = relpath(_dict['file'])
        return _dict

    def __str__(self):
        if False:
            return 10
        "\n        Creates a string representation of the SourceRange object.\n\n        If the whole file is affected, then just the filename is shown.\n\n        >>> str(SourceRange.from_values('test_file', None, None, None, None))\n        '...test_file'\n\n        If the whole line is affected, then just the filename with starting\n        line number and ending line number is shown.\n\n        >>> str(SourceRange.from_values('test_file', 1, None, 2, None))\n        '...test_file: L1 : L2'\n\n        This is the general case where particular column and line are\n        specified. It shows the starting line and column and ending line\n        and column, with filename in the beginning.\n\n        >>> str(SourceRange.from_values('test_file', 1, 1, 2, 1))\n        '...test_file: L1 C1 : L2 C1'\n        "
        if self.start.line is None and self.end.line is None:
            format_str = f'{self.start.file}'
        elif self.start.column is None and self.end.column is None:
            format_str = f'{self.start.file}: L{self.start.line} : L{self.end.line}'
        else:
            format_str = f'{self.start.file}: L{self.start.line} C{self.start.column} : L{self.end.line} C{self.end.column}'
        return format_str

    def overlaps(self, other):
        if False:
            while True:
                i = 10
        return self.start.file == other.start.file and super().overlaps(other)

    def __contains__(self, item):
        if False:
            i = 10
            return i + 15
        return super().__contains__(item) and self.start.file == item.start.file