import pytest
from vyper.exceptions import ImmutableViolation, InvalidType, TypeMismatch

def test_augassign(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    augassign_test = '\n@external\ndef augadd(x: int128, y: int128) -> int128:\n    z: int128 = x\n    z += y\n    return z\n\n@external\ndef augmul(x: int128, y: int128) -> int128:\n    z: int128 = x\n    z *= y\n    return z\n\n@external\ndef augsub(x: int128, y: int128) -> int128:\n    z: int128 = x\n    z -= y\n    return z\n\n@external\ndef augmod(x: int128, y: int128) -> int128:\n    z: int128 = x\n    z %= y\n    return z\n    '
    c = get_contract_with_gas_estimation(augassign_test)
    assert c.augadd(5, 12) == 17
    assert c.augmul(5, 12) == 60
    assert c.augsub(5, 12) == -7
    assert c.augmod(5, 12) == 5
    print('Passed aug-assignment test')

@pytest.mark.parametrize('typ,in_val,out_val', [('uint256', 77, 123), ('uint256[3]', [1, 2, 3], [4, 5, 6]), ('DynArray[uint256, 3]', [1, 2, 3], [4, 5, 6]), ('Bytes[5]', b'vyper', b'conda')])
def test_internal_assign(get_contract_with_gas_estimation, typ, in_val, out_val):
    if False:
        while True:
            i = 10
    code = f'\n@internal\ndef foo(x: {typ}) -> {typ}:\n    x = {out_val}\n    return x\n\n@external\ndef bar(x: {typ}) -> {typ}:\n    return self.foo(x)\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.bar(in_val) == out_val

def test_internal_assign_struct(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\nenum Bar:\n    BAD\n    BAK\n    BAZ\n\nstruct Foo:\n    a: uint256\n    b: DynArray[Bar, 3]\n    c: String[5]\n\n@internal\ndef foo(x: Foo) -> Foo:\n    x = Foo({a: 789, b: [Bar.BAZ, Bar.BAK, Bar.BAD], c: "conda"})\n    return x\n\n@external\ndef bar(x: Foo) -> Foo:\n    return self.foo(x)\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.bar((123, [1, 2, 4], 'vyper')) == (789, [4, 2, 1], 'conda')

def test_internal_assign_struct_member(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\nenum Bar:\n    BAD\n    BAK\n    BAZ\n\nstruct Foo:\n    a: uint256\n    b: DynArray[Bar, 3]\n    c: String[5]\n\n@internal\ndef foo(x: Foo) -> Foo:\n    x.a = 789\n    x.b.pop()\n    return x\n\n@external\ndef bar(x: Foo) -> Foo:\n    return self.foo(x)\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.bar((123, [1, 2, 4], 'vyper')) == (789, [1, 2], 'vyper')

def test_internal_augassign(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    code = '\n@internal\ndef foo(x: int128) -> int128:\n    x += 77\n    return x\n\n@external\ndef bar(x: int128) -> int128:\n    return self.foo(x)\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.bar(123) == 200

@pytest.mark.parametrize('typ', ['DynArray[uint256, 3]', 'uint256[3]'])
def test_internal_augassign_arrays(get_contract_with_gas_estimation, typ):
    if False:
        i = 10
        return i + 15
    code = f'\n@internal\ndef foo(x: {typ}) -> {typ}:\n    x[1] += 77\n    return x\n\n@external\ndef bar(x: {typ}) -> {typ}:\n    return self.foo(x)\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.bar([1, 2, 3]) == [1, 79, 3]

def test_invalid_external_assign(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\n@external\ndef foo(x: int128):\n    x = 5\n'
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code), ImmutableViolation)

def test_invalid_external_augassign(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef foo(x: int128):\n    x += 5\n'
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code), ImmutableViolation)

def test_valid_literal_increment(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\nstorx: uint256\n\n@external\ndef foo1() -> int128:\n    x: int128 = 122\n    x += 1\n    return x\n\n@external\ndef foo2() -> uint256:\n    x: uint256 = 122\n    x += 1\n    return x\n\n@external\ndef foo3(y: uint256) -> uint256:\n    self.storx = y\n    self.storx += 1\n    return self.storx\n'
    c = get_contract_with_gas_estimation(code)
    assert c.foo1() == 123
    assert c.foo2() == 123
    assert c.foo3(11) == 12

def test_invalid_uin256_assignment(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\nstorx: uint256\n\n@external\ndef foo2() -> uint256:\n    x: uint256 = -1\n    x += 1\n    return x\n'
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code), InvalidType)

def test_invalid_uin256_assignment_calculate_literals(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    code = '\nstorx: uint256\n\n@external\ndef foo2() -> uint256:\n    x: uint256 = 0\n    x = 3 * 4 / 2 + 1 - 2\n    return x\n'
    c = get_contract_with_gas_estimation(code)
    assert c.foo2() == 5

def test_nested_map_key_works(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    code = '\nstruct X:\n    a: int128\n    b: int128\nstruct Y:\n    c: int128\n    d: int128\ntest_map1: HashMap[int128, X]\ntest_map2: HashMap[int128, Y]\n\n@external\ndef set():\n    self.test_map1[1].a = 333\n    self.test_map2[333].c = 111\n\n\n@external\ndef get(i: int128) -> int128:\n    idx: int128 = self.test_map1[i].a\n    return self.test_map2[idx].c\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.set(transact={})
    assert c.get(1) == 111

def test_nested_map_key_problem(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\nstruct X:\n    a: int128\n    b: int128\nstruct Y:\n    c: int128\n    d: int128\ntest_map1: HashMap[int128, X]\ntest_map2: HashMap[int128, Y]\n\n@external\ndef set():\n    self.test_map1[1].a = 333\n    self.test_map2[333].c = 111\n\n\n@external\ndef get() -> int128:\n    return self.test_map2[self.test_map1[1].a].c\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.set(transact={})
    assert c.get() == 111

@pytest.mark.parametrize('contract', ['\n@external\ndef foo():\n    y: int128 = 1\n    z: decimal = y\n    ', '\n@external\ndef foo():\n    y: int128 = 1\n    z: decimal = 0.0\n    z = y\n    ', '\n@external\ndef foo():\n    y: bool = False\n    z: decimal = y\n    ', '\n@external\ndef foo():\n    y: bool = False\n    z: decimal = 0.0\n    z = y\n    ', '\n@external\ndef foo():\n    y: uint256 = 1\n    z: int128 = y\n    ', '\n@external\ndef foo():\n    y: uint256 = 1\n    z: int128 = 0\n    z = y\n    ', '\n@external\ndef foo():\n    y: int128 = 1\n    z: bytes32 = y\n    ', '\n@external\ndef foo():\n    y: int128 = 1\n    z: bytes32 = empty(bytes32)\n    z = y\n    ', '\n@external\ndef foo():\n    y: uint256 = 1\n    z: bytes32 = y\n    ', '\n@external\ndef foo():\n    y: uint256 = 1\n    z: bytes32 = empty(bytes32)\n    z = y\n    '])
def test_invalid_implicit_conversions(contract, assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    assert_compile_failed(lambda : get_contract_with_gas_estimation(contract), TypeMismatch)

def test_invalid_nonetype_assignment(assert_compile_failed, get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\n@internal\ndef bar():\n    pass\n\n@external\ndef foo():\n    ret : bool = self.bar()\n'
    assert_compile_failed(lambda : get_contract_with_gas_estimation(code), InvalidType)
overlap_codes = ['\n@external\ndef bug(xs: uint256[2]) -> uint256[2]:\n    # Initial value\n    ys: uint256[2] = xs\n    ys = [ys[1], ys[0]]\n    return ys\n    ', '\nfoo: uint256[2]\n@external\ndef bug(xs: uint256[2]) -> uint256[2]:\n    # Initial value\n    self.foo = xs\n    self.foo = [self.foo[1], self.foo[0]]\n    return self.foo\n    ']

@pytest.mark.parametrize('code', overlap_codes)
def test_assign_rhs_lhs_overlap(get_contract, code):
    if False:
        i = 10
        return i + 15
    c = get_contract(code)
    assert c.bug([1, 2]) == [2, 1]

def test_assign_rhs_lhs_partial_overlap(get_contract):
    if False:
        while True:
            i = 10
    code = '\n@external\ndef bug(xs: uint256[2]) -> uint256[2]:\n    # Initial value\n    ys: uint256[2] = xs\n    ys = [xs[1], ys[0]]\n    return ys\n    '
    c = get_contract(code)
    assert c.bug([1, 2]) == [2, 1]

def test_assign_rhs_lhs_overlap_dynarray(get_contract):
    if False:
        return 10
    code = '\n@external\ndef bug(xs: DynArray[uint256, 2]) -> DynArray[uint256, 2]:\n    ys: DynArray[uint256, 2] = xs\n    ys = [ys[1], ys[0]]\n    return ys\n    '
    c = get_contract(code)
    assert c.bug([1, 2]) == [2, 1]

def test_assign_rhs_lhs_overlap_struct(get_contract):
    if False:
        i = 10
        return i + 15
    code = '\nstruct Point:\n    x: uint256\n    y: uint256\n\n@external\ndef bug(p: Point) -> Point:\n    t: Point = p\n    t = Point({x: t.y, y: t.x})\n    return t\n    '
    c = get_contract(code)
    assert c.bug((1, 2)) == (2, 1)
mload_merge_codes = [('\n@external\ndef foo() -> uint256[4]:\n    # copy "backwards"\n    xs: uint256[4] = [1, 2, 3, 4]\n\n# dst < src\n    xs[0] = xs[1]\n    xs[1] = xs[2]\n    xs[2] = xs[3]\n\n    return xs\n    ', [2, 3, 4, 4]), ('\n@external\ndef foo() -> uint256[4]:\n    # copy "forwards"\n    xs: uint256[4] = [1, 2, 3, 4]\n\n# src < dst\n    xs[1] = xs[0]\n    xs[2] = xs[1]\n    xs[3] = xs[2]\n\n    return xs\n    ', [1, 1, 1, 1]), ('\n@external\ndef foo() -> uint256[5]:\n    # partial "forward" copy\n    xs: uint256[5] = [1, 2, 3, 4, 5]\n\n# src < dst\n    xs[2] = xs[0]\n    xs[3] = xs[1]\n    xs[4] = xs[2]\n\n    return xs\n    ', [1, 2, 1, 2, 1])]

@pytest.mark.parametrize('code,expected_result', mload_merge_codes)
def test_mcopy_overlap(get_contract, code, expected_result):
    if False:
        return 10
    c = get_contract(code)
    assert c.foo() == expected_result