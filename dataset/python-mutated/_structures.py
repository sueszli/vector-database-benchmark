class InfinityType:

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return 'Infinity'

    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        return False

    def __le__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        return False

    def __eq__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        return True

    def __ge__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        return True

    def __neg__(self: object) -> 'NegativeInfinityType':
        if False:
            return 10
        return NegativeInfinity
Infinity = InfinityType()

class NegativeInfinityType:

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '-Infinity'

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        return True

    def __le__(self, other: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        return isinstance(other, self.__class__)

    def __gt__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    def __ge__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        return False

    def __neg__(self: object) -> InfinityType:
        if False:
            while True:
                i = 10
        return Infinity
NegativeInfinity = NegativeInfinityType()