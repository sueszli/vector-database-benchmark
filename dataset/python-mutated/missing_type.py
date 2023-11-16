from builtins import _test_sink, _test_source

class UnknownSourceDef:

    def source(self) -> None:
        if False:
            print('Hello World!')
        pass
    unknown = source

def test_unknown_source_def(x: UnknownSourceDef) -> None:
    if False:
        for i in range(10):
            print('nop')
    y = x.unknown()
    _test_sink(y)

class UnknownSourceAttribute:

    def source(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass
    unknown = source

def test_unknown_source_attribute(x: UnknownSourceAttribute) -> None:
    if False:
        print('Hello World!')
    y = x.unknown()
    _test_sink(y)

class UnknownSinkDef:

    def sink(self, x: str) -> None:
        if False:
            while True:
                i = 10
        pass
    unknown = sink

def test_unknown_sink_def(x: UnknownSinkDef) -> None:
    if False:
        i = 10
        return i + 15
    x.unknown(_test_source())

class UnknownSinkAttribute:

    def sink(self, x: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass
    unknown = sink

def test_unknown_sink_attribute(x: UnknownSinkAttribute) -> None:
    if False:
        for i in range(10):
            print('nop')
    x.unknown(_test_source())