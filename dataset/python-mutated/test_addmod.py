def test_uint256_addmod(assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    uint256_code = '\n@external\ndef _uint256_addmod(x: uint256, y: uint256, z: uint256) -> uint256:\n    return uint256_addmod(x, y, z)\n    '
    c = get_contract_with_gas_estimation(uint256_code)
    assert c._uint256_addmod(1, 2, 2) == 1
    assert c._uint256_addmod(32, 2, 32) == 2
    assert c._uint256_addmod(2 ** 256 - 1, 0, 2) == 1
    assert c._uint256_addmod(2 ** 255, 2 ** 255, 6) == 4
    assert_tx_failed(lambda : c._uint256_addmod(1, 2, 0))

def test_uint256_addmod_ext_call(w3, side_effects_contract, assert_side_effects_invoked, get_contract):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef foo(f: Foo) -> uint256:\n    return uint256_addmod(32, 2, f.foo(32))\n\ninterface Foo:\n    def foo(x: uint256) -> uint256: payable\n    '
    c1 = side_effects_contract('uint256')
    c2 = get_contract(code)
    assert c2.foo(c1.address) == 2
    assert_side_effects_invoked(c1, lambda : c2.foo(c1.address, transact={}))

def test_uint256_addmod_internal_call(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\n@external\ndef foo() -> uint256:\n    return uint256_addmod(self.a(), self.b(), self.c())\n\n@internal\ndef a() -> uint256:\n    return 32\n\n@internal\ndef b() -> uint256:\n    return 2\n\n@internal\ndef c() -> uint256:\n    return 32\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo() == 2

def test_uint256_addmod_evaluation_order(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\na: uint256\n\n@external\ndef foo1() -> uint256:\n    self.a = 0\n    return uint256_addmod(self.a, 1, self.bar())\n\n@external\ndef foo2() -> uint256:\n    self.a = 0\n    return uint256_addmod(self.a, self.bar(), 3)\n\n@external\ndef foo3() -> uint256:\n    self.a = 0\n    return uint256_addmod(1, self.a, self.bar())\n\n@internal\ndef bar() -> uint256:\n    self.a = 1\n    return 2\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo1() == 1
    assert c.foo2() == 2
    assert c.foo3() == 1