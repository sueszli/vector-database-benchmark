from .helper import hopper

def test_sanity():
    if False:
        i = 10
        return i + 15
    data = hopper().tobytes()
    assert isinstance(data, bytes)