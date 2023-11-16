import binascii
import pytest
from cryptography.hazmat.primitives.asymmetric.x448 import X448PrivateKey, X448PublicKey
from .utils import wycheproof_tests

@pytest.mark.supported(only_if=lambda backend: backend.x448_supported(), skip_message='Requires OpenSSL with X448 support')
@wycheproof_tests('x448_test.json')
def test_x448(backend, wycheproof):
    if False:
        for i in range(10):
            print('nop')
    assert set(wycheproof.testgroup.items()) == {('curve', 'curve448'), ('type', 'XdhComp')}
    private_key = X448PrivateKey.from_private_bytes(binascii.unhexlify(wycheproof.testcase['private']))
    public_key_bytes = binascii.unhexlify(wycheproof.testcase['public'])
    if len(public_key_bytes) == 57:
        assert wycheproof.acceptable
        assert wycheproof.has_flag('NonCanonicalPublic')
        with pytest.raises(ValueError):
            X448PublicKey.from_public_bytes(public_key_bytes)
        return
    public_key = X448PublicKey.from_public_bytes(public_key_bytes)
    assert wycheproof.valid or wycheproof.acceptable
    expected = binascii.unhexlify(wycheproof.testcase['shared'])
    if expected == b'\x00' * 56:
        assert wycheproof.acceptable
        with pytest.raises(ValueError):
            private_key.exchange(public_key)
    else:
        assert private_key.exchange(public_key) == expected