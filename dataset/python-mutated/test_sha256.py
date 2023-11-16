import hashlib
import pytest
pytestmark = pytest.mark.usefixtures('memory_mocker')

def test_sha256_string_literal(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef bar() -> bytes32:\n    return sha256("test")\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.bar() == hashlib.sha256(b'test').digest()

def test_sha256_literal_bytes(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef bar() -> (bytes32 , bytes32):\n    x: bytes32 = sha256("test")\n    y: bytes32 = sha256(b"test")\n    return x, y\n    '
    c = get_contract_with_gas_estimation(code)
    h = hashlib.sha256(b'test').digest()
    assert c.bar() == [h, h]

def test_sha256_bytes32(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef bar(a: bytes32) -> bytes32:\n    return sha256(a)\n    '
    c = get_contract_with_gas_estimation(code)
    test_val = 8 * b'bBaA'
    assert c.bar(test_val) == hashlib.sha256(test_val).digest()

def test_sha256_bytearraylike(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\n@external\ndef bar(a: String[100]) -> bytes32:\n    return sha256(a)\n    '
    c = get_contract_with_gas_estimation(code)
    test_val = 'test me! test me!'
    assert c.bar(test_val) == hashlib.sha256(test_val.encode()).digest()
    test_val = 'fun'
    assert c.bar(test_val) == hashlib.sha256(test_val.encode()).digest()

def test_sha256_bytearraylike_storage(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\na: public(Bytes[100])\n\n@external\ndef set(b: Bytes[100]):\n    self.a = b\n\n@external\ndef bar() -> bytes32:\n    return sha256(self.a)\n    '
    c = get_contract_with_gas_estimation(code)
    test_val = b'test me! test me!'
    c.set(test_val, transact={})
    assert c.a() == test_val
    assert c.bar() == hashlib.sha256(test_val).digest()