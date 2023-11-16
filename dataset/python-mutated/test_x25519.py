import binascii
import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from .utils import wycheproof_tests

@pytest.mark.supported(only_if=lambda backend: backend.x25519_supported(), skip_message='Requires OpenSSL with X25519 support')
@wycheproof_tests('x25519_test.json')
def test_x25519(backend, wycheproof):
    if False:
        while True:
            i = 10
    assert set(wycheproof.testgroup.items()) == {('curve', 'curve25519'), ('type', 'XdhComp')}
    private_key = X25519PrivateKey.from_private_bytes(binascii.unhexlify(wycheproof.testcase['private']))
    public_key = X25519PublicKey.from_public_bytes(binascii.unhexlify(wycheproof.testcase['public']))
    assert wycheproof.valid or wycheproof.acceptable
    expected = binascii.unhexlify(wycheproof.testcase['shared'])
    if expected == b'\x00' * 32:
        assert wycheproof.acceptable
        with pytest.raises(ValueError):
            private_key.exchange(public_key)
    else:
        assert private_key.exchange(public_key) == expected