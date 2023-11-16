""" Source code reference record.

All the information to lookup line and file of a code location, together with
the future flags in use there.
"""
from nuitka.__past__ import total_ordering
from nuitka.utils.InstanceCounters import counted_del, counted_init, isCountingInstances

@total_ordering
class SourceCodeReference(object):
    __slots__ = ['filename', 'line', 'column']

    @classmethod
    def fromFilenameAndLine(cls, filename, line):
        if False:
            print('Hello World!')
        result = cls()
        result.filename = filename
        result.line = line
        return result
    if isCountingInstances():
        __del__ = counted_del()

    @counted_init
    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.filename = None
        self.line = None
        self.column = None

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<%s to %s:%s>' % (self.__class__.__name__, self.filename, self.line)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((self.filename, self.line, self.column))

    def __lt__(self, other):
        if False:
            print('Hello World!')
        if other is None:
            return True
        if other is self:
            return False
        assert isinstance(other, SourceCodeReference), other
        if self.filename < other.filename:
            return True
        elif self.filename > other.filename:
            return False
        elif self.line < other.line:
            return True
        elif self.line > other.line:
            return False
        elif self.column < other.column:
            return True
        elif self.column > other.column:
            return False
        else:
            return self.isInternal() < other.isInternal()

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if other is None:
            return False
        if other is self:
            return True
        assert isinstance(other, SourceCodeReference), other
        if self.filename != other.filename:
            return False
        if self.line != other.line:
            return False
        if self.column != other.column:
            return False
        return self.isInternal() is other.isInternal()

    def _clone(self, line):
        if False:
            return 10
        'Make a copy it itself.'
        return self.fromFilenameAndLine(filename=self.filename, line=line)

    def atInternal(self):
        if False:
            while True:
                i = 10
        'Make a copy it itself but mark as internal code.\n\n        Avoids useless copies, by returning an internal object again if\n        it is already internal.\n        '
        if not self.isInternal():
            result = self._clone(self.line)
            return result
        else:
            return self

    def atLineNumber(self, line):
        if False:
            return 10
        'Make a reference to the same file, but different line.\n\n        Avoids useless copies, by returning same object if the line is\n        the same.\n        '
        assert type(line) is int, line
        if self.line != line:
            return self._clone(line)
        else:
            return self

    def atColumnNumber(self, column):
        if False:
            i = 10
            return i + 15
        assert type(column) is int, column
        if self.column != column:
            result = self._clone(self.line)
            result.column = column
            return result
        else:
            return self

    def getLineNumber(self):
        if False:
            return 10
        return self.line

    def getColumnNumber(self):
        if False:
            i = 10
            return i + 15
        return self.column

    def getFilename(self):
        if False:
            while True:
                i = 10
        return self.filename

    def getAsString(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s:%s' % (self.filename, self.line)

    @staticmethod
    def isInternal():
        if False:
            return 10
        return False

class SourceCodeReferenceInternal(SourceCodeReference):
    __slots__ = ()

    @staticmethod
    def isInternal():
        if False:
            i = 10
            return i + 15
        return True

def fromFilename(filename):
    if False:
        while True:
            i = 10
    return SourceCodeReference.fromFilenameAndLine(filename=filename, line=1)