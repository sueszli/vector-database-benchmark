from coalib.results.TextPosition import TextPosition
from coala_utils.decorators import enforce_signature

class AbsolutePosition(TextPosition):

    @enforce_signature
    def __init__(self, text: (tuple, list, None)=None, position: (int, None)=None):
        if False:
            while True:
                i = 10
        '\n        Creates an AbsolutePosition object that represents the index of a\n        character in a string.\n\n        :param text:     The text containing the character.\n        :param position: Position identifying the index of character\n                         in text.\n        '
        line = column = None
        if position is not None and text is not None:
            (line, column) = calc_line_col(text, position)
        self._text = text
        self._position = position
        super().__init__(line, column)

    @property
    def position(self):
        if False:
            i = 10
            return i + 15
        return self._position

def calc_line_col(text, position):
    if False:
        i = 10
        return i + 15
    "\n    Creates a tuple containing (line, column) by calculating line number\n    and column in the text, from position.\n\n    The position represents the index of a character. In the following\n    example 'a' is at position '0' and it's corresponding line and column are:\n\n    >>> calc_line_col(('a\\n',), 0)\n    (1, 1)\n\n    All special characters(including the newline character) belong in the same\n    line, and have their own position. A line is an item in the tuple:\n\n    >>> calc_line_col(('a\\n', 'b\\n'), 1)\n    (1, 2)\n    >>> calc_line_col(('a\\n', 'b\\n'), 2)\n    (2, 1)\n\n    :param text:          A tuple/list of lines in which position is to\n                          be calculated.\n    :param position:      Position (starting from 0) of character to be found\n                          in the (line, column) form.\n    :return:              A tuple of the form (line, column), where both line\n                          and column start from 1.\n    "
    for (linenum, line) in enumerate(text, start=1):
        linelen = len(line)
        if position < linelen:
            return (linenum, position + 1)
        position -= linelen
    raise ValueError('Position not found in text')