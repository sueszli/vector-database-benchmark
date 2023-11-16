"""A simple file parser that can tell whether the first character of a line
is quoted or not"""
from .languages import _COMMENT

class StringParser:
    """A simple file parser that can tell whether the first character of a line
    is quoted or not"""
    single = None
    triple = None
    triple_start = None

    def __init__(self, language):
        if False:
            for i in range(10):
                print('nop')
        self.ignore = language is None
        self.python = language != 'R'
        self.comment = _COMMENT.get(language)

    def is_quoted(self):
        if False:
            return 10
        'Is the next line quoted?'
        if self.ignore:
            return False
        return self.single or self.triple

    def read_line(self, line):
        if False:
            while True:
                i = 10
        'Read a new line'
        if self.ignore:
            return
        if not self.is_quoted() and self.comment is not None and line.lstrip().startswith(self.comment):
            return
        self.triple_start = -1
        for (i, char) in enumerate(line):
            if self.single is None and self.triple is None and self.comment and self.comment.startswith(char) and line[i:].startswith(self.comment):
                break
            if char not in ['"', "'"]:
                continue
            if line[i - 1:i] == '\\':
                continue
            if self.single == char:
                self.single = None
                continue
            if self.single is not None:
                continue
            if not self.python:
                continue
            if line[i - 2:i + 1] == 3 * char and i >= self.triple_start + 3:
                if self.triple == char:
                    self.triple = None
                    self.triple_start = i
                    continue
                if self.triple is not None:
                    continue
                self.triple = char
                self.triple_start = i
                continue
            if self.triple is not None:
                continue
            self.single = char
        if self.python:
            self.single = None