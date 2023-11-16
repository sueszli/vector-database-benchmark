from os.path import relpath, abspath
from coala_utils.decorators import enforce_signature, generate_ordering, generate_repr, get_public_members
from coalib.results.TextPosition import TextPosition

@generate_repr('file', 'line', 'column')
@generate_ordering('file', 'line', 'column')
class SourcePosition(TextPosition):

    @enforce_signature
    def __init__(self, file: str, line=None, column=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new result position object that represents the position of a\n        result in the source code.\n\n        :param file:        The filename.\n        :param line:        The line in file or None, the first line is 1.\n        :param column:      The column indicating the character. The first one\n                            in a line is 1.\n        :raises TypeError:  Raised when\n                            - file is not a string or None.\n                            - line or columns are no integers.\n        '
        TextPosition.__init__(self, line, column)
        self.filename = file
        self._file = abspath(file)

    @property
    def file(self):
        if False:
            print('Hello World!')
        return self._file

    def __json__(self, use_relpath=False):
        if False:
            for i in range(10):
                print('nop')
        _dict = get_public_members(self)
        if use_relpath:
            _dict['file'] = relpath(_dict['file'])
        return _dict

    def __str__(self):
        if False:
            return 10
        source_position = self.filename
        if self.line is not None:
            source_position += ':' + str(self.line)
        if self.column is not None:
            source_position += ':' + str(self.column)
        return source_position