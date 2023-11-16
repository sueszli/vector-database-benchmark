class IntervalSet:

    @classmethod
    def from_string(cls, s):
        if False:
            print('Hello World!')
        "Return a tuple of intervals, covering the codepoints of characters in `s`.\n\n        >>> IntervalSet.from_string('abcdef0123456789')\n        ((48, 57), (97, 102))\n        "
        x = cls(((ord(c), ord(c)) for c in sorted(s)))
        return x.union(x)

    def __init__(self, intervals):
        if False:
            i = 10
            return i + 15
        self.intervals = tuple(intervals)
        self.offsets = [0]
        for (u, v) in self.intervals:
            self.offsets.append(self.offsets[-1] + v - u + 1)
        self.size = self.offsets.pop()

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self.size

    def __iter__(self):
        if False:
            while True:
                i = 10
        for (u, v) in self.intervals:
            yield from range(u, v + 1)

    def __getitem__(self, i):
        if False:
            i = 10
            return i + 15
        if i < 0:
            i = self.size + i
        if i < 0 or i >= self.size:
            raise IndexError(f'Invalid index {i} for [0, {self.size})')
        j = len(self.intervals) - 1
        if self.offsets[j] > i:
            hi = j
            lo = 0
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if self.offsets[mid] <= i:
                    lo = mid
                else:
                    hi = mid
            j = lo
        t = i - self.offsets[j]
        (u, v) = self.intervals[j]
        r = u + t
        assert r <= v
        return r

    def __contains__(self, elem):
        if False:
            while True:
                i = 10
        if isinstance(elem, str):
            elem = ord(elem)
        assert isinstance(elem, int)
        assert 0 <= elem <= 1114111
        return any((start <= elem <= end for (start, end) in self.intervals))

    def __repr__(self):
        if False:
            return 10
        return f'IntervalSet({self.intervals!r})'

    def index(self, value):
        if False:
            for i in range(10):
                print('nop')
        for (offset, (u, v)) in zip(self.offsets, self.intervals):
            if u == value:
                return offset
            elif u > value:
                raise ValueError(f'{value} is not in list')
            if value <= v:
                return offset + (value - u)
        raise ValueError(f'{value} is not in list')

    def index_above(self, value):
        if False:
            return 10
        for (offset, (u, v)) in zip(self.offsets, self.intervals):
            if u >= value:
                return offset
            if value <= v:
                return offset + (value - u)
        return self.size

    def __or__(self, other):
        if False:
            i = 10
            return i + 15
        return self.union(other)

    def __sub__(self, other):
        if False:
            return 10
        return self.difference(other)

    def __and__(self, other):
        if False:
            while True:
                i = 10
        return self.intersection(other)

    def union(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Merge two sequences of intervals into a single tuple of intervals.\n\n        Any integer bounded by `x` or `y` is also bounded by the result.\n\n        >>> union([(3, 10)], [(1, 2), (5, 17)])\n        ((1, 17),)\n        '
        assert isinstance(other, type(self))
        x = self.intervals
        y = other.intervals
        if not x:
            return IntervalSet(((u, v) for (u, v) in y))
        if not y:
            return IntervalSet(((u, v) for (u, v) in x))
        intervals = sorted(x + y, reverse=True)
        result = [intervals.pop()]
        while intervals:
            (u, v) = intervals.pop()
            (a, b) = result[-1]
            if u <= b + 1:
                result[-1] = (a, max(v, b))
            else:
                result.append((u, v))
        return IntervalSet(result)

    def difference(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Set difference for lists of intervals. That is, returns a list of\n        intervals that bounds all values bounded by x that are not also bounded by\n        y. x and y are expected to be in sorted order.\n\n        For example difference([(1, 10)], [(2, 3), (9, 15)]) would\n        return [(1, 1), (4, 8)], removing the values 2, 3, 9 and 10 from the\n        interval.\n        '
        assert isinstance(other, type(self))
        x = self.intervals
        y = other.intervals
        if not y:
            return IntervalSet(x)
        x = list(map(list, x))
        i = 0
        j = 0
        result = []
        while i < len(x) and j < len(y):
            (xl, xr) = x[i]
            assert xl <= xr
            (yl, yr) = y[j]
            assert yl <= yr
            if yr < xl:
                j += 1
            elif yl > xr:
                result.append(x[i])
                i += 1
            elif yl <= xl:
                if yr >= xr:
                    i += 1
                else:
                    x[i][0] = yr + 1
                    j += 1
            else:
                result.append((xl, yl - 1))
                if yr + 1 <= xr:
                    x[i][0] = yr + 1
                    j += 1
                else:
                    i += 1
        result.extend(x[i:])
        return IntervalSet(map(tuple, result))

    def intersection(self, other):
        if False:
            return 10
        'Set intersection for lists of intervals.'
        assert isinstance(other, type(self)), other
        intervals = []
        i = j = 0
        while i < len(self.intervals) and j < len(other.intervals):
            (u, v) = self.intervals[i]
            (U, V) = other.intervals[j]
            if u > V:
                j += 1
            elif U > v:
                i += 1
            else:
                intervals.append((max(u, U), min(v, V)))
                if v < V:
                    i += 1
                else:
                    j += 1
        return IntervalSet(intervals)