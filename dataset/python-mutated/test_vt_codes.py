import io
from mitmproxy.utils.vt_codes import ensure_supported

def test_simple():
    if False:
        i = 10
        return i + 15
    assert not ensure_supported(io.StringIO())