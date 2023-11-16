from io import StringIO
from antlr4.Token import Token
IntervalSet = None

class IntervalSet(object):
    __slots__ = ('intervals', 'readonly')

    def __init__(self):
        if False:
            print('Hello World!')
        self.intervals = None
        self.readonly = False

    def __iter__(self):
        if False:
            while True:
                i = 10
        if self.intervals is not None:
            for i in self.intervals:
                for c in i:
                    yield c

    def __getitem__(self, item):
        if False:
            return 10
        i = 0
        for k in self:
            if i == item:
                return k
            else:
                i += 1
        return Token.INVALID_TYPE

    def addOne(self, v: int):
        if False:
            print('Hello World!')
        self.addRange(range(v, v + 1))

    def addRange(self, v: range):
        if False:
            return 10
        if self.intervals is None:
            self.intervals = list()
            self.intervals.append(v)
        else:
            k = 0
            for i in self.intervals:
                if v.stop < i.start:
                    self.intervals.insert(k, v)
                    return
                elif v.stop == i.start:
                    self.intervals[k] = range(v.start, i.stop)
                    return
                elif v.start <= i.stop:
                    self.intervals[k] = range(min(i.start, v.start), max(i.stop, v.stop))
                    self.reduce(k)
                    return
                k += 1
            self.intervals.append(v)

    def addSet(self, other: IntervalSet):
        if False:
            print('Hello World!')
        if other.intervals is not None:
            for i in other.intervals:
                self.addRange(i)
        return self

    def reduce(self, k: int):
        if False:
            for i in range(10):
                print('nop')
        if k < len(self.intervals) - 1:
            l = self.intervals[k]
            r = self.intervals[k + 1]
            if l.stop >= r.stop:
                self.intervals.pop(k + 1)
                self.reduce(k)
            elif l.stop >= r.start:
                self.intervals[k] = range(l.start, r.stop)
                self.intervals.pop(k + 1)

    def complement(self, start, stop):
        if False:
            return 10
        result = IntervalSet()
        result.addRange(range(start, stop + 1))
        for i in self.intervals:
            result.removeRange(i)
        return result

    def __contains__(self, item):
        if False:
            return 10
        if self.intervals is None:
            return False
        else:
            return any((item in i for i in self.intervals))

    def __len__(self):
        if False:
            return 10
        return sum((len(i) for i in self.intervals))

    def removeRange(self, v):
        if False:
            while True:
                i = 10
        if v.start == v.stop - 1:
            self.removeOne(v.start)
        elif self.intervals is not None:
            k = 0
            for i in self.intervals:
                if v.stop <= i.start:
                    return
                elif v.start > i.start and v.stop < i.stop:
                    self.intervals[k] = range(i.start, v.start)
                    x = range(v.stop, i.stop)
                    self.intervals.insert(k, x)
                    return
                elif v.start <= i.start and v.stop >= i.stop:
                    self.intervals.pop(k)
                    k -= 1
                elif v.start < i.stop:
                    self.intervals[k] = range(i.start, v.start)
                elif v.stop < i.stop:
                    self.intervals[k] = range(v.stop, i.stop)
                k += 1

    def removeOne(self, v):
        if False:
            for i in range(10):
                print('nop')
        if self.intervals is not None:
            k = 0
            for i in self.intervals:
                if v < i.start:
                    return
                elif v == i.start and v == i.stop - 1:
                    self.intervals.pop(k)
                    return
                elif v == i.start:
                    self.intervals[k] = range(i.start + 1, i.stop)
                    return
                elif v == i.stop - 1:
                    self.intervals[k] = range(i.start, i.stop - 1)
                    return
                elif v < i.stop - 1:
                    x = range(i.start, v)
                    self.intervals[k] = range(v + 1, i.stop)
                    self.intervals.insert(k, x)
                    return
                k += 1

    def toString(self, literalNames: list, symbolicNames: list):
        if False:
            while True:
                i = 10
        if self.intervals is None:
            return '{}'
        with StringIO() as buf:
            if len(self) > 1:
                buf.write('{')
            first = True
            for i in self.intervals:
                for j in i:
                    if not first:
                        buf.write(', ')
                    buf.write(self.elementName(literalNames, symbolicNames, j))
                    first = False
            if len(self) > 1:
                buf.write('}')
            return buf.getvalue()

    def elementName(self, literalNames: list, symbolicNames: list, a: int):
        if False:
            i = 10
            return i + 15
        if a == Token.EOF:
            return '<EOF>'
        elif a == Token.EPSILON:
            return '<EPSILON>'
        else:
            if a < len(literalNames) and literalNames[a] != '<INVALID>':
                return literalNames[a]
            if a < len(symbolicNames):
                return symbolicNames[a]
            return '<UNKNOWN>'