import copy
from coala_utils.decorators import enforce_signature, generate_ordering, generate_repr
from coalib.results.TextPosition import TextPosition

@generate_repr('start', 'end')
@generate_ordering('start', 'end')
class TextRange:

    @enforce_signature
    def __init__(self, start: TextPosition, end: (TextPosition, None)=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Creates a new TextRange.\n\n        :param start:       A TextPosition indicating the start of the range.\n                            Can't be ``None``.\n        :param end:         A TextPosition indicating the end of the range. If\n                            ``None`` is given, the start object will be used\n                            here.\n        :raises TypeError:  Raised when\n                            - start is not of type TextPosition.\n                            - end is neither of type TextPosition, nor is it\n                              None.\n        :raises ValueError: Raised when end position is smaller than start\n                            position, because negative ranges are not allowed.\n        "
        self._start = start
        self._end = copy.deepcopy(start) if end is None else end
        if self._end < start:
            raise ValueError("End position can't be less than start position.")

    @classmethod
    def from_values(cls, start_line=None, start_column=None, end_line=None, end_column=None):
        if False:
            print('Hello World!')
        '\n        Creates a new TextRange.\n\n        :param start_line:   The line number of the start position. The first\n                             line is 1.\n        :param start_column: The column number of the start position. The first\n                             column is 1.\n        :param end_line:     The line number of the end position. If this\n                             parameter is ``None``, then the end position is set\n                             the same like start position and end_column gets\n                             ignored.\n        :param end_column:   The column number of the end position.\n        :return:             A TextRange.\n        '
        start = TextPosition(start_line, start_column)
        if end_line is None:
            end = None
        else:
            end = TextPosition(end_line, end_column)
        return cls(start, end)

    @classmethod
    def join(cls, a, b):
        if False:
            print('Hello World!')
        '\n        Creates a new TextRange that covers the area of two overlapping ones\n\n        :param a: TextRange (needs to overlap b)\n        :param b: TextRange (needs to overlap a)\n        :return:  A new TextRange covering the union of the Area of a and b\n        '
        if not isinstance(a, cls) or not isinstance(b, cls):
            raise TypeError(f'only instances of {cls.__name__} can be joined')
        if not a.overlaps(b):
            raise ValueError(f'{cls.__name__}s must overlap to be joined')
        return cls(min(a.start, b.start), max(a.end, b.end))

    @property
    def start(self):
        if False:
            while True:
                i = 10
        return self._start

    @property
    def end(self):
        if False:
            for i in range(10):
                print('nop')
        return self._end

    def overlaps(self, other):
        if False:
            print('Hello World!')
        return self.start <= other.end and self.end >= other.start

    def expand(self, text_lines):
        if False:
            for i in range(10):
                print('nop')
        '\n        Passes a new TextRange that covers the same area of a file as this one\n        would. All values of None get replaced with absolute values.\n\n        values of None will be interpreted as follows:\n        self.start.line is None:   -> 1\n        self.start.column is None: -> 1\n        self.end.line is None:     -> last line of file\n        self.end.column is None:   -> last column of self.end.line\n\n        :param text_lines: File contents of the applicable file\n        :return:           TextRange with absolute values\n        '
        start_line = 1 if self.start.line is None else self.start.line
        start_column = 1 if self.start.column is None else self.start.column
        end_line = len(text_lines) if self.end.line is None else self.end.line
        end_column = len(text_lines[end_line - 1]) if self.end.column is None else self.end.column
        return TextRange.from_values(start_line, start_column, end_line, end_column)

    def __contains__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return item.start >= self.start and item.end <= self.end