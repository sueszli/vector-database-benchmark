from builtins import _test_sink
'\nThis test demonstrates a case where the analysis is exponential due to 4\nmethods recursively calling each other with different paths. The set of sinks\nwould potentially be infinite, it is all combinations of [a], [b], [c] with an\nending [x], i.e `([a]|[b]|[c])+[x]` in a regular expression. This is prevented\nusing the widening on the taint tree, which limits the maximum depth of the\ntree.\n'

class Base:

    def method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

class A(Base):

    def __init__(self, base: Base) -> None:
        if False:
            while True:
                i = 10
        self.a: Base = base

    def method(self):
        if False:
            return 10
        self.a.method()

class B(Base):

    def __init__(self, base: Base) -> None:
        if False:
            i = 10
            return i + 15
        self.b: Base = base

    def method(self):
        if False:
            return 10
        self.b.method()

class C(Base):

    def __init__(self, base: Base, x: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.c: Base = base
        self.x: str = x

    def method(self):
        if False:
            return 10
        _test_sink(self.x)
        self.c.method()