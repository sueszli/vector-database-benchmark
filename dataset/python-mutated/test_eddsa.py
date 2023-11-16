import binascii
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed448 import Ed448PublicKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from .utils import wycheproof_tests

@pytest.mark.supported(only_if=lambda backend: backend.ed25519_supported(), skip_message='Requires OpenSSL with Ed25519 support')
@wycheproof_tests('eddsa_test.json')
def test_ed25519_signature(backend, wycheproof):
    if False:
        return 10
    assert wycheproof.testgroup['key']['curve'] == 'edwards25519'
    key = Ed25519PublicKey.from_public_bytes(binascii.unhexlify(wycheproof.testgroup['key']['pk']))
    if wycheproof.valid or wycheproof.acceptable:
        key.verify(binascii.unhexlify(wycheproof.testcase['sig']), binascii.unhexlify(wycheproof.testcase['msg']))
    else:
        with pytest.raises(InvalidSignature):
            key.verify(binascii.unhexlify(wycheproof.testcase['sig']), binascii.unhexlify(wycheproof.testcase['msg']))

@pytest.mark.supported(only_if=lambda backend: backend.ed448_supported(), skip_message='Requires OpenSSL with Ed448 support')
@wycheproof_tests('ed448_test.json')
def test_ed448_signature(backend, wycheproof):
    if False:
        print('Hello World!')
    key = Ed448PublicKey.from_public_bytes(binascii.unhexlify(wycheproof.testgroup['key']['pk']))
    if wycheproof.valid or wycheproof.acceptable:
        key.verify(binascii.unhexlify(wycheproof.testcase['sig']), binascii.unhexlify(wycheproof.testcase['msg']))
    else:
        with pytest.raises(InvalidSignature):
            key.verify(binascii.unhexlify(wycheproof.testcase['sig']), binascii.unhexlify(wycheproof.testcase['msg']))