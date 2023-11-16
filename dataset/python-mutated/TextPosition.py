from coala_utils.decorators import enforce_signature, generate_ordering, generate_repr

class ZeroOffsetError(ValueError):
    pass

@generate_repr('line', 'column')
@generate_ordering('line', 'column')
class TextPosition:

    @enforce_signature
    def __init__(self, line: (int, None)=None, column: (int, None)=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new TextPosition object that represents the position inside\n        a string with line/column numbers.\n\n        :param line:        The line in file or None, the first line is 1.\n        :param column:      The column indicating the character. The first one\n                            in a line is 1.\n        :raises TypeError:  Raised when line or columns are no integers.\n        :raises ValueError: Raised when a column is set but line is None.\n        '
        if line is None and column is not None:
            raise ValueError('A column can only be set if a line is set.')
        if line == 0 and column == 0:
            raise ZeroOffsetError('Line and column offset cannot be zero.')
        elif line == 0:
            raise ZeroOffsetError('Line offset cannot be zero.')
        elif column == 0:
            raise ZeroOffsetError('Column offset cannot be zero.')
        self._line = line
        self._column = column

    @property
    def line(self):
        if False:
            for i in range(10):
                print('nop')
        return self._line

    @property
    def column(self):
        if False:
            while True:
                i = 10
        return self._column

    def __le__(self, other):
        if False:
            return 10
        '\n        Test whether ``self`` is behind or equals the other\n        ``TextPosition``.\n\n        If the column in a ``TextPosition`` is ``None``, consider\n        whole line. If the line in a ``TextPosition`` is ``None``,\n        consider whole file.\n\n        :param other: ``TextPosition`` to compare with.\n        :return:      Whether this ``TextPosition`` is behind the other\n                      one or the same.\n        '
        if self.line is None or other.line is None:
            return True
        if self.line == other.line:
            return True if self.column is None or other.column is None else self.column <= other.column
        else:
            return self.line < other.line

    def __ge__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test whether ``self`` is ahead of or equals the\n        other ``TextPosition``.\n\n        If the column in a ``TextPosition`` is ``None``, consider\n        whole line. If the line in a ``TextPosition`` is ``None``,\n        consider whole file.\n\n        :param other: ``TextPosition`` to compare with.\n        :return:      Whether this ``TextPosition`` is ahead of the other\n                      one or the same.\n        '
        if self.line is None or other.line is None:
            return True
        if self.line == other.line:
            return True if self.column is None or other.column is None else self.column >= other.column
        else:
            return self.line > other.line