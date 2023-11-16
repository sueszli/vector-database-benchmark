class Foo:
    __slots__ = 'bar'

    def __init__(self, bar):
        if False:
            while True:
                i = 10
        self.bar = bar

class Foo:
    __slots__: str = 'bar'

    def __init__(self, bar):
        if False:
            i = 10
            return i + 15
        self.bar = bar

class Foo:
    __slots__: str = f'bar'

    def __init__(self, bar):
        if False:
            print('Hello World!')
        self.bar = bar

class Foo:
    __slots__ = ('bar',)

    def __init__(self, bar):
        if False:
            i = 10
            return i + 15
        self.bar = bar

class Foo:
    __slots__: tuple[str, ...] = ('bar',)

    def __init__(self, bar):
        if False:
            for i in range(10):
                print('nop')
        self.bar = bar