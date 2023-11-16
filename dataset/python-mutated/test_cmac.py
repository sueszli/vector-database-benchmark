import binascii
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.ciphers.algorithms import AES
from cryptography.hazmat.primitives.cmac import CMAC
from .utils import wycheproof_tests

@wycheproof_tests('aes_cmac_test.json')
def test_aes_cmac(backend, wycheproof):
    if False:
        for i in range(10):
            print('nop')
    key = binascii.unhexlify(wycheproof.testcase['key'])
    msg = binascii.unhexlify(wycheproof.testcase['msg'])
    tag = binascii.unhexlify(wycheproof.testcase['tag'])
    if wycheproof.valid and len(tag) == 16:
        ctx = CMAC(AES(key), backend)
        ctx.update(msg)
        ctx.verify(tag)
    elif len(key) not in [16, 24, 32]:
        with pytest.raises(ValueError):
            CMAC(AES(key), backend)
    else:
        ctx = CMAC(AES(key), backend)
        ctx.update(msg)
        with pytest.raises(InvalidSignature):
            ctx.verify(tag)