from builtins import _test_sink, _test_source
from dataclasses import dataclass

class TitoAttributes:

    def __init__(self, x, y, z):
        if False:
            while True:
                i = 10
        self.x = x
        self.y = y
        self.z = z

def test_tito_attribute_x():
    if False:
        i = 10
        return i + 15
    c = TitoAttributes(**_test_source())
    _test_sink(c.x)

def test_tito_attribute_y():
    if False:
        i = 10
        return i + 15
    c = TitoAttributes(**_test_source())
    _test_sink(c.y)

def test_tito_attribute_z_with_tag():
    if False:
        while True:
            i = 10
    c = TitoAttributes(**_test_source())
    _test_sink(c.z)

def test_tito_attribute_join():
    if False:
        for i in range(10):
            print('nop')
    c = TitoAttributes(**_test_source())
    foo = c.x
    if 1:
        foo = c.y
    elif 2:
        foo = c.z
    _test_sink(foo)

@dataclass
class SourceAttributes:
    x: str = ''
    y: str = ''
    z: str = ''

def test_source_attribute_x(c: SourceAttributes):
    if False:
        print('Hello World!')
    _test_sink(c.x)

def test_source_attribute_y(c: SourceAttributes):
    if False:
        for i in range(10):
            print('nop')
    _test_sink(c.y)

def test_source_attribute_z(c: SourceAttributes):
    if False:
        while True:
            i = 10
    _test_sink(c.z)

def test_source_attribute_join(c: SourceAttributes):
    if False:
        i = 10
        return i + 15
    foo = c.x
    if 1:
        foo = c.y
    elif 2:
        foo = c.z
    _test_sink(foo)

@dataclass
class SinkAttributes:
    x: str = ''
    y: str = ''
    z: str = ''

def test_sink_attribute_x(c: SinkAttributes):
    if False:
        return 10
    c.x = _test_source()

def test_sink_attribute_y(c: SinkAttributes):
    if False:
        i = 10
        return i + 15
    c.y = _test_source()

def test_sink_attribute_z(c: SinkAttributes):
    if False:
        for i in range(10):
            print('nop')
    c.z = _test_source()

@dataclass
class TitoAttributeModelQuery:
    x: str = ''
    y: str = ''
    z: str = ''

def test_tito_attribute_model_query_x():
    if False:
        return 10
    _test_sink(TitoAttributeModelQuery(x=_test_source(), y='', z=''))

def test_tito_attribute_model_query_y():
    if False:
        for i in range(10):
            print('nop')
    _test_sink(TitoAttributeModelQuery(x='', y=_test_source(), z=''))

def test_tito_attribute_model_query_z():
    if False:
        return 10
    _test_sink(TitoAttributeModelQuery(x='', y='', z=_test_source()))