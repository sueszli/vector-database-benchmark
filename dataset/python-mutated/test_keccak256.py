def test_hash_code(get_contract_with_gas_estimation, keccak):
    if False:
        print('Hello World!')
    hash_code = '\n@external\ndef foo(inp: Bytes[100]) -> bytes32:\n    return keccak256(inp)\n\n@external\ndef foob() -> bytes32:\n    return keccak256(b"inp")\n\n@external\ndef bar() -> bytes32:\n    return keccak256("inp")\n    '
    c = get_contract_with_gas_estimation(hash_code)
    for inp in (b'', b'cow', b's' * 31, b'\xff' * 32, b'\n' * 33, b'g' * 64, b'h' * 65):
        assert '0x' + c.foo(inp).hex() == keccak(inp).hex()
    assert '0x' + c.bar().hex() == keccak(b'inp').hex()
    assert '0x' + c.foob().hex() == keccak(b'inp').hex()

def test_hash_code2(get_contract_with_gas_estimation):
    if False:
        return 10
    hash_code2 = '\n@external\ndef foo(inp: Bytes[100]) -> bool:\n    return keccak256(inp) == keccak256("badminton")\n    '
    c = get_contract_with_gas_estimation(hash_code2)
    assert c.foo(b'badminto') is False
    assert c.foo(b'badminton') is True

def test_hash_code3(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    hash_code3 = '\ntest: Bytes[100]\n\n@external\ndef set_test(inp: Bytes[100]):\n    self.test = inp\n\n@external\ndef tryy(inp: Bytes[100]) -> bool:\n    return keccak256(inp) == keccak256(self.test)\n\n@external\ndef tryy_str(inp: String[100]) -> bool:\n    return keccak256(inp) == keccak256(self.test)\n\n@external\ndef trymem(inp: Bytes[100]) -> bool:\n    x: Bytes[100] = self.test\n    return keccak256(inp) == keccak256(x)\n\n@external\ndef try32(inp: bytes32) -> bool:\n    return keccak256(inp) == keccak256(self.test)\n\n    '
    c = get_contract_with_gas_estimation(hash_code3)
    c.set_test(b'', transact={})
    assert c.tryy(b'') is True
    assert c.tryy_str('') is True
    assert c.trymem(b'') is True
    assert c.tryy(b'cow') is False
    c.set_test(b'cow', transact={})
    assert c.tryy(b'') is False
    assert c.tryy(b'cow') is True
    assert c.tryy_str('cow') is True
    c.set_test(b'5' * 32, transact={})
    assert c.tryy(b'5' * 32) is True
    assert c.trymem(b'5' * 32) is True
    assert c.try32(b'5' * 32) is True
    assert c.tryy(b'5' * 33) is False
    c.set_test(b'5' * 33, transact={})
    assert c.tryy(b'5' * 32) is False
    assert c.trymem(b'5' * 32) is False
    assert c.try32(b'5' * 32) is False
    assert c.tryy(b'5' * 33) is True
    print('Passed KECCAK256 hash test')