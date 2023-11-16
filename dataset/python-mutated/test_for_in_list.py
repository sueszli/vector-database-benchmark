from decimal import Decimal
import pytest
from vyper.exceptions import ArgumentException, ImmutableViolation, InvalidType, IteratorException, NamespaceCollision, StateAccessViolation, StructureException, TypeMismatch
BASIC_FOR_LOOP_CODE = [('\n@external\ndef data() -> int128:\n    s: int128[5] = [1, 2, 3, 4, 5]\n    for i in s:\n        if i >= 3:\n            return i\n    return -1', 3), ('\n@external\ndef data() -> int128:\n    s: DynArray[int128, 10] = [1, 2, 3, 4, 5]\n    for i in s:\n        if i >= 3:\n            return i\n    return -1', 3), ('\nstruct S:\n    x: int128\n    y: int128\n\n@external\ndef data() -> int128:\n    sss: DynArray[DynArray[S, 10], 10] = [\n        [S({x:1, y:2})],\n        [S({x:3, y:4}), S({x:5, y:6}), S({x:7, y:8}), S({x:9, y:10})]\n        ]\n    ret: int128 = 0\n    for ss in sss:\n        for s in ss:\n            ret += s.x + s.y\n    return ret', sum(range(1, 11))), ('\n@external\ndef data() -> int128:\n    for i in [3, 5, 7, 9]:\n        if i > 5:\n            return i\n    return -1', 7), ('\n@external\ndef data() -> String[33]:\n    xs: DynArray[String[33], 3] = ["hello", ",", "world"]\n    for x in xs:\n        if x == ",":\n            return x\n    return ""\n    ', ','), ('\n@external\ndef data() -> String[33]:\n    for x in ["hello", ",", "world"]:\n        if x == ",":\n            return x\n    return ""\n    ', ','), ('\n@external\ndef data() -> DynArray[String[33], 2]:\n    for x in [["hello", "world"], ["goodbye", "world!"]]:\n        if x[1] == "world":\n            return x\n    return []\n    ', ['hello', 'world']), ('\n@external\ndef data() -> int128:\n    ret: int128 = 0\n    xss: int128[3][3] = [[1,2,3],[4,5,6],[7,8,9]]\n    for xs in xss:\n        for x in xs:\n            ret += x\n    return ret', sum(range(1, 10))), ('\nstruct S:\n    x: int128\n    y: int128\n\n@external\ndef data() -> int128:\n    ret: int128 = 0\n    for ss in [[S({x:1, y:2})]]:\n        for s in ss:\n            ret += s.x + s.y\n    return ret', 1 + 2), ('\n@external\ndef data() -> address:\n    addresses: address[3] = [\n        0x7d577a597B2742b498Cb5Cf0C26cDCD726d39E6e,\n        0x82A978B3f5962A5b0957d9ee9eEf472EE55B42F1,\n        0xDCEceAF3fc5C0a63d195d69b1A90011B7B19650D\n    ]\n    count: int128 = 0\n    for i in addresses:\n        count += 1\n        if count == 2:\n            return i\n    return 0x0000000000000000000000000000000000000000\n    ', '0x82A978B3f5962A5b0957d9ee9eEf472EE55B42F1')]

@pytest.mark.parametrize('code, data', BASIC_FOR_LOOP_CODE)
def test_basic_for_in_lists(code, data, get_contract):
    if False:
        return 10
    c = get_contract(code)
    assert c.data() == data

def test_basic_for_list_storage(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\nx: int128[4]\n\n@external\ndef set():\n    self.x = [3, 5, 7, 9]\n\n@external\ndef data() -> int128:\n    for i in self.x:\n        if i > 5:\n            return i\n    return -1\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.data() == -1
    c.set(transact={})
    assert c.data() == 7

def test_basic_for_dyn_array_storage(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\nx: DynArray[int128, 4]\n\n@external\ndef set(xs: DynArray[int128, 4]):\n    self.x = xs\n\n@external\ndef data() -> int128:\n    t: int128 = 0\n    for i in self.x:\n        t += i\n    return t\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.data() == 0
    for xs in [[3, 5, 7, 9], [4, 6, 8], [1, 2], [5], []]:
        c.set(xs, transact={})
        assert c.data() == sum(xs)

def test_basic_for_list_storage_address(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\naddresses: address[3]\n\n@external\ndef set(i: int128, val: address):\n    self.addresses[i] = val\n\n@external\ndef ret(i: int128) -> address:\n    return self.addresses[i]\n\n@external\ndef iterate_return_second() -> address:\n    count: int128 = 0\n    for i in self.addresses:\n        count += 1\n        if count == 2:\n            return i\n    return empty(address)\n    '
    c = get_contract_with_gas_estimation(code)
    c.set(0, '0x82A978B3f5962A5b0957d9ee9eEf472EE55B42F1', transact={})
    c.set(1, '0x7d577a597B2742b498Cb5Cf0C26cDCD726d39E6e', transact={})
    c.set(2, '0xDCEceAF3fc5C0a63d195d69b1A90011B7B19650D', transact={})
    assert c.ret(1) == c.iterate_return_second() == '0x7d577a597B2742b498Cb5Cf0C26cDCD726d39E6e'

def test_basic_for_list_storage_decimal(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\nreadings: decimal[3]\n\n@external\ndef set(i: int128, val: decimal):\n    self.readings[i] = val\n\n@external\ndef ret(i: int128) -> decimal:\n    return self.readings[i]\n\n@external\ndef i_return(break_count: int128) -> decimal:\n    count: int128 = 0\n    for i in self.readings:\n        if count == break_count:\n            return i\n        count += 1\n    return -1.111\n    '
    c = get_contract_with_gas_estimation(code)
    c.set(0, Decimal('0.0001'), transact={})
    c.set(1, Decimal('1.1'), transact={})
    c.set(2, Decimal('2.2'), transact={})
    assert c.ret(2) == c.i_return(2) == Decimal('2.2')
    assert c.ret(1) == c.i_return(1) == Decimal('1.1')
    assert c.ret(0) == c.i_return(0) == Decimal('0.0001')

def test_for_in_list_iter_type(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\n@view\ndef func(amounts: uint256[3]) -> uint256:\n    total: uint256 = as_wei_value(0, "wei")\n\n    # calculate total\n    for amount in amounts:\n        total += amount\n\n    return total\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.func([100, 200, 300]) == 600

def test_for_in_dyn_array(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\n@view\ndef func(amounts: DynArray[uint256, 3]) -> uint256:\n    total: uint256 = 0\n\n    # calculate total\n    for amount in amounts:\n        total += amount\n\n    return total\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.func([100, 200, 300]) == 600
    assert c.func([100, 200]) == 300
GOOD_CODE = ['\n@external\ndef foo(x: int128):\n    p: int128 = 0\n    for i in range(3):\n        p += i\n    for i in range(4):\n        p += i\n    ', '\n@external\ndef foo(x: int128):\n    p: int128 = 0\n    for i in range(3):\n        p += i\n    for i in [1, 2, 3, 4]:\n        p += i\n    ', '\n@external\ndef foo(x: int128):\n    p: int128 = 0\n    for i in [1, 2, 3, 4]:\n        p += i\n    for i in [1, 2, 3, 4]:\n        p += i\n    ', '\n@external\ndef foo():\n    for i in range(10):\n        pass\n    for i in range(20):\n        pass\n    ', '\n@external\ndef foo():\n    for i in range(10):\n        pass\n    i: int128 = 100  # create new variable i\n    i = 200  # look up the variable i and check whether it is in forvars\n    ']

@pytest.mark.parametrize('code', GOOD_CODE)
def test_good_code(code, get_contract):
    if False:
        i = 10
        return i + 15
    get_contract(code)
RANGE_CONSTANT_CODE = [('\nTREE_FIDDY: constant(int128)  = 350\n\n\n@external\ndef a() -> uint256:\n    x: uint256 = 0\n    for i in range(TREE_FIDDY):\n        x += 1\n    return x', 350), ('\nONE_HUNDRED: constant(int128)  = 100\n\n@external\ndef a() -> uint256:\n    x: uint256 = 0\n    for i in range(1, 1 + ONE_HUNDRED):\n        x += 1\n    return x', 100), ('\nSTART: constant(int128)  = 100\nEND: constant(int128)  = 199\n\n@external\ndef a() -> uint256:\n    x: uint256 = 0\n    for i in range(START, END):\n        x += 1\n    return x', 99), ('\n@external\ndef a() -> int128:\n    x: int128 = 0\n    for i in range(-5, -1):\n        x += i\n    return x', -14)]

@pytest.mark.parametrize('code, result', RANGE_CONSTANT_CODE)
def test_range_constant(get_contract, code, result):
    if False:
        i = 10
        return i + 15
    c = get_contract(code)
    assert c.a() == result
BAD_CODE = [('\n@external\ndef data() -> int128:\n    s: int128[6] = [1, 2, 3, 4, 5, 6]\n    count: int128 = 0\n    for i in s:\n        s[count] = 1  # this should not be allowed.\n        if i >= 3:\n            return i\n        count += 1\n    return -1\n    ', ImmutableViolation), ('\n@external\ndef foo():\n    s: int128[6] = [1, 2, 3, 4, 5, 6]\n    count: int128 = 0\n    for i in s:\n        s[count] += 1\n    ', ImmutableViolation), ('\ns: int128[6]\n\n@external\ndef set():\n    self.s = [1, 2, 3, 4, 5, 6]\n\n@external\ndef data() -> int128:\n    count: int128 = 0\n    for i in self.s:\n        self.s[count] = 1  # this should not be allowed.\n        if i >= 3:\n            return i\n        count += 1\n    return -1\n    ', ImmutableViolation), ('\nstruct Foo:\n    foo: uint256[4]\n\nmy_array2: Foo\n\n@internal\ndef doStuff(i: uint256) -> uint256:\n    self.my_array2.foo[i] = i\n    return i\n\n@internal\ndef _helper():\n    i: uint256 = 0\n    for item in self.my_array2.foo:\n        self.doStuff(i)\n        i += 1\n    ', ImmutableViolation), ('\nstruct Foo:\n    foo: uint256[4]\n\nstruct Bar:\n    bar: Foo\n    baz: uint256\n\nmy_array2: Bar\n\n@internal\ndef doStuff(i: uint256) -> uint256:\n    self.my_array2.bar.foo[i] = i\n    return i\n\n@internal\ndef _helper():\n    i: uint256 = 0\n    for item in self.my_array2.bar.foo:\n        self.doStuff(i)\n        i += 1\n    ', ImmutableViolation), ('\nstruct Foo:\n    foo: uint256[4]\n\nmy_array2: Foo\n\n@internal\ndef doStuff():\n    self.my_array2.foo = [\n        block.timestamp + 1,\n        block.timestamp + 2,\n        block.timestamp + 3,\n        block.timestamp + 4\n    ]\n\n@internal\ndef _helper():\n    i: uint256 = 0\n    for item in self.my_array2.foo:\n        self.doStuff()\n        i += 1\n    ', ImmutableViolation), ('\n@external\ndef foo(x: int128):\n    for i in range(4):\n        for i in range(5):\n            pass\n    ', NamespaceCollision), ('\n@external\ndef foo(x: int128):\n    for i in [1,2]:\n        for i in [1,2]:\n            pass\n     ', NamespaceCollision), ('\n@external\ndef foo(x: int128):\n    for i in [1,2]:\n        i = 2\n    ', ImmutableViolation), ('\n@external\ndef foo():\n    xs: DynArray[uint256, 5] = [1,2,3]\n    for x in xs:\n        xs.pop()\n    ', ImmutableViolation), ('\n@external\ndef foo():\n    xs: DynArray[uint256, 5] = [1,2,3]\n    for x in xs:\n        xs.append(x)\n    ', ImmutableViolation), ('\n@external\ndef foo():\n    xs: DynArray[DynArray[uint256, 5], 5] = [[1,2,3]]\n    for x in xs:\n        x.pop()\n    ', ImmutableViolation), ('\narray: DynArray[uint256, 5]\n@internal\ndef a():\n    self.b()\n\n@internal\ndef b():\n    self.array.pop()\n\n@external\ndef foo():\n    for x in self.array:\n        self.a()\n    ', ImmutableViolation), ('\n@external\ndef foo(x: int128):\n    for i in [1,2]:\n        i += 2\n    ', ImmutableViolation), ('\n@external\ndef foo():\n    for i in range(-3):\n        pass\n    ', StructureException), '\n@external\ndef foo():\n    for i in range(0):\n        pass\n    ', '\n@external\ndef foo():\n    for i in []:\n        pass\n    ', '\nFOO: constant(DynArray[uint256, 3]) = []\n\n@external\ndef foo():\n    for i in FOO:\n        pass\n    ', ('\n@external\ndef foo():\n    for i in range(5,3):\n        pass\n    ', StructureException), ('\n@external\ndef foo():\n    for i in range(5,3,-1):\n        pass\n    ', ArgumentException), ('\n@external\ndef foo():\n    a: uint256 = 2\n    for i in range(a):\n        pass\n    ', StateAccessViolation), '\n@external\ndef foo():\n    a: int128 = 6\n    for i in range(a,a-3):\n        pass\n    ', ('\n@external\ndef foo():\n    for i in range():\n        pass\n    ', ArgumentException), ('\n@external\ndef foo():\n    for i in range(0,1,2):\n        pass\n    ', ArgumentException), ('\n@external\ndef foo():\n    for i in b"asdf":\n        pass\n    ', InvalidType), ('\n@external\ndef foo():\n    for i in 31337:\n        pass\n    ', InvalidType), ('\n@external\ndef foo():\n    for i in bar():\n        pass\n    ', IteratorException), ('\n@external\ndef foo():\n    for i in self.bar():\n        pass\n    ', IteratorException), ('\n@external\ndef test_for() -> int128:\n    a: int128 = 0\n    for i in range(max_value(int128), max_value(int128)+2):\n        a = i\n    return a\n    ', TypeMismatch), ('\n@external\ndef test_for() -> int128:\n    a: int128 = 0\n    b: uint256 = 0\n    for i in range(5):\n        a = i\n        b = i\n    return a\n    ', TypeMismatch)]

@pytest.mark.parametrize('code', BAD_CODE)
def test_bad_code(assert_compile_failed, get_contract, code):
    if False:
        while True:
            i = 10
    err = StructureException
    if not isinstance(code, str):
        (code, err) = code
    assert_compile_failed(lambda : get_contract(code), err)