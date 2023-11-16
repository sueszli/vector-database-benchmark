def test_uint256_mulmod(assert_tx_failed, get_contract_with_gas_estimation):
    if False:
        return 10
    uint256_code = '\n@external\ndef _uint256_mulmod(x: uint256, y: uint256, z: uint256) -> uint256:\n    return uint256_mulmod(x, y, z)\n    '
    c = get_contract_with_gas_estimation(uint256_code)
    assert c._uint256_mulmod(3, 1, 2) == 1
    assert c._uint256_mulmod(200, 3, 601) == 600
    assert c._uint256_mulmod(2 ** 255, 1, 3) == 2
    assert c._uint256_mulmod(2 ** 255, 2, 6) == 4
    assert_tx_failed(lambda : c._uint256_mulmod(2, 2, 0))

def test_uint256_mulmod_complex(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    modexper = '\n@external\ndef exponential(base: uint256, exponent: uint256, modulus: uint256) -> uint256:\n    o: uint256 = 1\n    for i in range(256):\n        o = uint256_mulmod(o, o, modulus)\n        if exponent & shift(1, 255 - i) != 0:\n            o = uint256_mulmod(o, base, modulus)\n    return o\n    '
    c = get_contract_with_gas_estimation(modexper)
    assert c.exponential(3, 5, 100) == 43
    assert c.exponential(2, 997, 997) == 2

def test_uint256_mulmod_ext_call(w3, side_effects_contract, assert_side_effects_invoked, get_contract):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef foo(f: Foo) -> uint256:\n    return uint256_mulmod(200, 3, f.foo(601))\n\ninterface Foo:\n    def foo(x: uint256) -> uint256: nonpayable\n    '
    c1 = side_effects_contract('uint256')
    c2 = get_contract(code)
    assert c2.foo(c1.address) == 600
    assert_side_effects_invoked(c1, lambda : c2.foo(c1.address, transact={}))

def test_uint256_mulmod_internal_call(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\n@external\ndef foo() -> uint256:\n    return uint256_mulmod(self.a(), self.b(), self.c())\n\n@internal\ndef a() -> uint256:\n    return 200\n\n@internal\ndef b() -> uint256:\n    return 3\n\n@internal\ndef c() -> uint256:\n    return 601\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo() == 600

def test_uint256_mulmod_evaluation_order(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\na: uint256\n\n@external\ndef foo1() -> uint256:\n    self.a = 1\n    return uint256_mulmod(self.a, 2, self.bar())\n\n@external\ndef foo2() -> uint256:\n    self.a = 1\n    return uint256_mulmod(self.bar(), self.a, 2)\n\n@external\ndef foo3() -> uint256:\n    self.a = 1\n    return uint256_mulmod(2, self.a, self.bar())\n\n@internal\ndef bar() -> uint256:\n    self.a = 7\n    return 5\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo1() == 2
    assert c.foo2() == 1
    assert c.foo3() == 2