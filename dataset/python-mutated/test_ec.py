G1 = [1, 2]
G1_times_two = [1368015179489954701390400359078579693043519447331113978918064868415326638035, 9918110051302171585080402603319702774565515993150576347155970296011118125764]
G1_times_three = [3353031288059533942658390886683067124040920775575537747144343083137631628272, 19321533766552368860946552437480515441416830039777911637913418824951667761761]
negative_G1 = [1, 21888242871839275222246405745257275088696311157297823662689037894645226208581]
curve_order = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def test_ecadd(get_contract_with_gas_estimation):
    if False:
        return 10
    ecadder = '\nx3: uint256[2]\ny3: uint256[2]\n\n@external\ndef _ecadd(x: uint256[2], y: uint256[2]) -> uint256[2]:\n    return ecadd(x, y)\n\n@external\ndef _ecadd2(x: uint256[2], y: uint256[2]) -> uint256[2]:\n    x2: uint256[2] = x\n    y2: uint256[2] = [y[0], y[1]]\n    return ecadd(x2, y2)\n\n@external\ndef _ecadd3(x: uint256[2], y: uint256[2]) -> uint256[2]:\n    self.x3 = x\n    self.y3 = [y[0], y[1]]\n    return ecadd(self.x3, self.y3)\n\n    '
    c = get_contract_with_gas_estimation(ecadder)
    assert c._ecadd(G1, G1) == G1_times_two
    assert c._ecadd2(G1, G1_times_two) == G1_times_three
    assert c._ecadd3(G1, [0, 0]) == G1
    assert c._ecadd3(G1, negative_G1) == [0, 0]

def test_ecadd_internal_call(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\n@internal\ndef a() -> uint256[2]:\n    return [1, 2]\n\n@external\ndef foo() -> uint256[2]:\n    return ecadd([1, 2], self.a())\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo() == G1_times_two

def test_ecadd_ext_call(w3, side_effects_contract, assert_side_effects_invoked, get_contract):
    if False:
        i = 10
        return i + 15
    code = '\ninterface Foo:\n    def foo(x: uint256[2]) -> uint256[2]: payable\n\n@external\ndef foo(a: Foo) -> uint256[2]:\n    return ecadd([1, 2], a.foo([1, 2]))\n    '
    c1 = side_effects_contract('uint256[2]')
    c2 = get_contract(code)
    assert c2.foo(c1.address) == G1_times_two
    assert_side_effects_invoked(c1, lambda : c2.foo(c1.address, transact={}))

def test_ecadd_evaluation_order(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\nx: uint256[2]\n\n@internal\ndef bar() -> uint256[2]:\n    self.x = ecadd([1, 2], [1, 2])\n    return [1, 2]\n\n@external\ndef foo() -> bool:\n    self.x = [1, 2]\n    a: uint256[2] = ecadd([1, 2], [1, 2])\n    b: uint256[2] = ecadd(self.x, self.bar())\n    return a[0] == b[0] and a[1] == b[1]\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo() is True

def test_ecmul(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    ecmuller = '\nx3: uint256[2]\ny3: uint256\n\n@external\ndef _ecmul(x: uint256[2], y: uint256) -> uint256[2]:\n    return ecmul(x, y)\n\n@external\ndef _ecmul2(x: uint256[2], y: uint256) -> uint256[2]:\n    x2: uint256[2] = x\n    y2: uint256 = y\n    return ecmul(x2, y2)\n\n@external\ndef _ecmul3(x: uint256[2], y: uint256) -> uint256[2]:\n    self.x3 = x\n    self.y3 = y\n    return ecmul(self.x3, self.y3)\n\n'
    c = get_contract_with_gas_estimation(ecmuller)
    assert c._ecmul(G1, 0) == [0, 0]
    assert c._ecmul(G1, 1) == G1
    assert c._ecmul(G1, 3) == G1_times_three
    assert c._ecmul(G1, curve_order - 1) == negative_G1
    assert c._ecmul(G1, curve_order) == [0, 0]

def test_ecmul_internal_call(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    code = '\n@internal\ndef a() -> uint256:\n    return 3\n\n@external\ndef foo() -> uint256[2]:\n    return ecmul([1, 2], self.a())\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo() == G1_times_three

def test_ecmul_ext_call(w3, side_effects_contract, assert_side_effects_invoked, get_contract):
    if False:
        return 10
    code = '\ninterface Foo:\n    def foo(x: uint256) -> uint256: payable\n\n@external\ndef foo(a: Foo) -> uint256[2]:\n    return ecmul([1, 2], a.foo(3))\n    '
    c1 = side_effects_contract('uint256')
    c2 = get_contract(code)
    assert c2.foo(c1.address) == G1_times_three
    assert_side_effects_invoked(c1, lambda : c2.foo(c1.address, transact={}))

def test_ecmul_evaluation_order(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\nx: uint256[2]\n\n@internal\ndef bar() -> uint256:\n    self.x = ecmul([1, 2], 3)\n    return 3\n\n@external\ndef foo() -> bool:\n    self.x = [1, 2]\n    a: uint256[2] = ecmul([1, 2], 3)\n    b: uint256[2] = ecmul(self.x, self.bar())\n    return a[0] == b[0] and a[1] == b[1]\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.foo() is True