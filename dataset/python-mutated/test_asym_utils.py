import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.utils import Prehashed, decode_dss_signature, encode_dss_signature

def test_dss_signature():
    if False:
        for i in range(10):
            print('nop')
    sig = encode_dss_signature(1, 1)
    assert sig == b'0\x06\x02\x01\x01\x02\x01\x01'
    assert decode_dss_signature(sig) == (1, 1)
    r_s1 = (1037234182290683143945502320610861668562885151617, 559776156650501990899426031439030258256861634312)
    sig2 = encode_dss_signature(*r_s1)
    assert sig2 == b'0-\x02\x15\x00\xb5\xaf0xg\xfb\x8bT9\x00\x13\xccg\x02\r\xdf\x1f,\x0b\x81\x02\x14b\r;"\xabP1D\x0c>5\xea\xb6\xf4\x81)\x8f\x9e\x9f\x08'
    assert decode_dss_signature(sig2) == r_s1
    sig3 = encode_dss_signature(0, 0)
    assert sig3 == b'0\x06\x02\x01\x00\x02\x01\x00'
    assert decode_dss_signature(sig3) == (0, 0)

def test_encode_dss_non_integer():
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError):
        encode_dss_signature('h', 3)
    with pytest.raises(TypeError):
        encode_dss_signature('3', '2')
    with pytest.raises(TypeError):
        encode_dss_signature(3, 'h')
    with pytest.raises(TypeError):
        encode_dss_signature(3.3, 1.2)
    with pytest.raises(TypeError):
        encode_dss_signature('hello', 'world')

def test_encode_dss_negative():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        encode_dss_signature(-1, 0)
    with pytest.raises(ValueError):
        encode_dss_signature(0, -1)

def test_decode_dss_trailing_bytes():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        decode_dss_signature(b'0\x06\x02\x01\x01\x02\x01\x01\x00\x00\x00')

def test_decode_dss_invalid_asn1():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):
        decode_dss_signature(b'0\x07\x02\x01\x01\x02\x02\x01')
    with pytest.raises(ValueError):
        decode_dss_signature(b'\x00\x00')

def test_pass_invalid_prehashed_arg():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError):
        Prehashed(object())

def test_prehashed_digest_size():
    if False:
        print('Hello World!')
    p = Prehashed(hashes.SHA256())
    assert p.digest_size == 32