from functools import total_ordering

@total_ordering
class MutInt:
    __slots__ = ['value']

    def __init__(self, value):
        if False:
            print('Hello World!')
        self.value = value

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self.value)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'MutInt({self.value!r})'

    def __format__(self, fmt):
        if False:
            return 10
        return format(self.value, fmt)

    def __add__(self, other):
        if False:
            return 10
        if isinstance(other, MutInt):
            return MutInt(self.value + other.value)
        elif isinstance(other, int):
            return MutInt(self.value + other)
        else:
            return NotImplemented
    __radd__ = __add__

    def __iadd__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, MutInt):
            self.value += other.value
            return self
        elif isinstance(other, int):
            self.value += other
            return self
        else:
            return NotImplemented

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, MutInt):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        else:
            return NotImplemented

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, MutInt):
            return self.value < other.value
        elif isinstance(other, int):
            return self.value < other
        else:
            return NotImplemented

    def __int__(self):
        if False:
            i = 10
            return i + 15
        return int(self.value)

    def __float__(self):
        if False:
            i = 10
            return i + 15
        return float(self.value)
    __index__ = __int__