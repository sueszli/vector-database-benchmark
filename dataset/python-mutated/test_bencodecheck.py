import pytest
from tribler.core.utilities.bencodecheck import is_bencoded

def test_bencode_checker():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError, match='^Value should be of bytes type. Got: str$'):
        is_bencoded('3:abc')
    assert not is_bencoded(b'')
    assert not is_bencoded(b'3:abc3:abc')
    assert not is_bencoded(b'3:abce')
    assert is_bencoded(b'0:')
    assert is_bencoded(b'3:abc')
    assert not is_bencoded(b'03:abc')
    assert not is_bencoded(b'4:abc')
    assert not is_bencoded(b'3abc')
    assert is_bencoded(b'i0e')
    assert is_bencoded(b'i123e')
    assert is_bencoded(b'i-123e')
    assert not is_bencoded(b'i0123e')
    assert not is_bencoded(b'i00e')
    assert not is_bencoded(b'i-0e')
    assert not is_bencoded(b'i-00e')
    assert not is_bencoded(b'i-0123e')
    assert is_bencoded(b'de')
    assert is_bencoded(b'd3:abc3:defe')
    assert not is_bencoded(b'd3:abce')
    assert not is_bencoded(b'd3:abc3:def')
    assert not is_bencoded(b'di123e3:defe')
    assert is_bencoded(b'd3:abcd3:foo3:baree')
    assert is_bencoded(b'le')
    assert is_bencoded(b'li123e3:abcd3:foo3:barelee')
    assert is_bencoded(b'lli123e3:abceli456e3:defee')
    assert not is_bencoded(b'l3:abc')
    assert not is_bencoded(b'hello')
    assert not is_bencoded(b'<?=#.')