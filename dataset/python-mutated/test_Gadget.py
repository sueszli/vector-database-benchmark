import pytest

@pytest.fixture
def gadget():
    if False:
        for i in range(10):
            print('nop')
    import plugin_api as pa
    g = pa.Gadget()
    return g

def test_creation(gadget):
    if False:
        i = 10
        return i + 15
    pass

def test_property(gadget):
    if False:
        for i in range(10):
            print('nop')
    gadget.prop = 42
    assert gadget.prop == 42

def test_push(gadget):
    if False:
        for i in range(10):
            print('nop')
    gadget.push(42)