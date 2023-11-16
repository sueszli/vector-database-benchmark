import io
import pytest
from falcon.stream import BoundedStream

@pytest.fixture
def bounded_stream():
    if False:
        while True:
            i = 10
    return BoundedStream(io.BytesIO(), 1024)

def test_not_writable(bounded_stream):
    if False:
        i = 10
        return i + 15
    assert not bounded_stream.writable()
    with pytest.raises(IOError):
        bounded_stream.write(b'something something')