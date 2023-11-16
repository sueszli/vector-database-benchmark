class SortedSet:

    def __init__(self, iterable=None, key=None):
        if False:
            return 10
        self._key = key if key is not None else lambda x: x
        self._set = set(iterable) if iterable is not None else set()
        self._cached_last = None
        self._cached_first = None

    def first(self):
        if False:
            while True:
                i = 10
        if self._cached_first is not None:
            return self._cached_first
        first = None
        for element in self._set:
            if first is None or self._key(first) > self._key(element):
                first = element
        self._cached_first = first
        return first

    def last(self):
        if False:
            i = 10
            return i + 15
        if self._cached_last is not None:
            return self._cached_last
        last = None
        for element in self._set:
            if last is None or self._key(last) < self._key(element):
                last = element
        self._cached_last = last
        return last

    def pop_last(self):
        if False:
            i = 10
            return i + 15
        value = self.last()
        self._set.remove(value)
        self._cached_last = None
        return value

    def add(self, value):
        if False:
            print('Hello World!')
        if self._cached_last is not None and self._key(value) > self._key(self._cached_last):
            self._cached_last = value
        if self._cached_first is not None and self._key(value) < self._key(self._cached_first):
            self._cached_first = value
        return self._set.add(value)

    def remove(self, value):
        if False:
            while True:
                i = 10
        if self._cached_last is not None and self._cached_last == value:
            self._cached_last = None
        if self._cached_first is not None and self._cached_first == value:
            self._cached_first = None
        return self._set.remove(value)

    def __contains__(self, value):
        if False:
            for i in range(10):
                print('nop')
        return value in self._set

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(sorted(self._set, key=self._key))

    def _bool(self):
        if False:
            return 10
        return len(self._set) != 0
    __nonzero__ = _bool
    __bool__ = _bool