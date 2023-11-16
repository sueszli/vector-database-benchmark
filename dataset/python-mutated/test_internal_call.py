import string
from decimal import Decimal
import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from vyper.compiler import compile_code
from vyper.exceptions import ArgumentException, CallViolation
pytestmark = pytest.mark.usefixtures('memory_mocker')

def test_selfcall_code(get_contract_with_gas_estimation):
    if False:
        return 10
    selfcall_code = '\n@internal\ndef _foo() -> int128:\n    return 3\n\n@external\ndef bar() -> int128:\n    return self._foo()\n    '
    c = get_contract_with_gas_estimation(selfcall_code)
    assert c.bar() == 3
    print('Passed no-argument self-call test')

def test_selfcall_code_2(get_contract_with_gas_estimation, keccak):
    if False:
        i = 10
        return i + 15
    selfcall_code_2 = '\n@internal\ndef _double(x: int128) -> int128:\n    return x * 2\n\n@external\ndef returnten() -> int128:\n    return self._double(5)\n\n@internal\ndef _hashy(x: bytes32) -> bytes32:\n    return keccak256(x)\n\n@external\ndef return_hash_of_rzpadded_cow() -> bytes32:\n    return self._hashy(0x636f770000000000000000000000000000000000000000000000000000000000)\n    '
    c = get_contract_with_gas_estimation(selfcall_code_2)
    assert c.returnten() == 10
    assert c.return_hash_of_rzpadded_cow() == keccak(b'cow' + b'\x00' * 29)
    print('Passed single fixed-size argument self-call test')

def test_selfcall_optimizer(get_contract):
    if False:
        print('Hello World!')
    code = '\ncounter: uint256\n\n@internal\ndef increment_counter() -> uint256:\n    self.counter += 1\n    return self.counter\n@external\ndef foo() -> (uint256, uint256):\n    x: uint256 = unsafe_mul(self.increment_counter(), 0)\n    return x, self.counter\n    '
    c = get_contract(code)
    assert c.foo() == [0, 1]

def test_selfcall_code_3(get_contract_with_gas_estimation, keccak):
    if False:
        for i in range(10):
            print('nop')
    selfcall_code_3 = '\n@internal\ndef _hashy2(x: Bytes[100]) -> bytes32:\n    return keccak256(x)\n\n@external\ndef return_hash_of_cow_x_30() -> bytes32:\n    return self._hashy2(b"cowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcowcow")  # noqa: E501\n\n@internal\ndef _len(x: Bytes[100]) -> uint256:\n    return len(x)\n\n@external\ndef returnten() -> uint256:\n    return self._len(b"badminton!")\n    '
    c = get_contract_with_gas_estimation(selfcall_code_3)
    assert c.return_hash_of_cow_x_30() == keccak(b'cow' * 30)
    assert c.returnten() == 10
    print('Passed single variable-size argument self-call test')

def test_selfcall_code_4(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    selfcall_code_4 = '\n@internal\ndef _summy(x: int128, y: int128) -> int128:\n    return x + y\n\n@internal\ndef _catty(x: Bytes[5], y: Bytes[5]) -> Bytes[10]:\n    return concat(x, y)\n\n@internal\ndef _slicey1(x: Bytes[10], y: uint256) -> Bytes[10]:\n    return slice(x, 0, y)\n\n@internal\ndef _slicey2(y: uint256, x: Bytes[10]) -> Bytes[10]:\n    return slice(x, 0, y)\n\n@external\ndef returnten() -> int128:\n    return self._summy(3, 7)\n\n@external\ndef return_mongoose() -> Bytes[10]:\n    return self._catty(b"mon", b"goose")\n\n@external\ndef return_goose() -> Bytes[10]:\n    return self._slicey1(b"goosedog", 5)\n\n@external\ndef return_goose2() -> Bytes[10]:\n    return self._slicey2(5, b"goosedog")\n    '
    c = get_contract_with_gas_estimation(selfcall_code_4)
    assert c.returnten() == 10
    assert c.return_mongoose() == b'mongoose'
    assert c.return_goose() == b'goose'
    assert c.return_goose2() == b'goose'
    print('Passed multi-argument self-call test')

def test_selfcall_code_5(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    selfcall_code_5 = '\ncounter: int128\n\n@internal\ndef _increment():\n    self.counter += 1\n\n@external\ndef returnten() -> int128:\n    for i in range(10):\n        self._increment()\n    return self.counter\n    '
    c = get_contract_with_gas_estimation(selfcall_code_5)
    assert c.returnten() == 10
    print('Passed self-call statement test')

def test_selfcall_code_6(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    selfcall_code_6 = '\nexcls: Bytes[32]\n\n@internal\ndef _set_excls(arg: Bytes[32]):\n    self.excls = arg\n\n@internal\ndef _underscore() -> Bytes[1]:\n    return b"_"\n\n@internal\ndef _hardtest(x: Bytes[100], y: uint256, z: uint256, a: Bytes[100], b: uint256, c: uint256) -> Bytes[201]:  # noqa: E501\n    return concat(slice(x, y, z), self._underscore(), slice(a, b, c))\n\n@external\ndef return_mongoose_revolution_32_excls() -> Bytes[201]:\n    self._set_excls(b"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")\n    return self._hardtest(b"megamongoose123", 4, 8, concat(b"russian revolution", self.excls), 8, 42)\n    '
    c = get_contract_with_gas_estimation(selfcall_code_6)
    assert c.return_mongoose_revolution_32_excls() == b'mongoose_revolution' + b'!' * 32
    print('Passed composite self-call test')

def test_list_call(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\n@internal\ndef _foo0(x: int128[2]) -> int128:\n    return x[0]\n\n@internal\ndef _foo1(x: int128[2]) -> int128:\n    return x[1]\n\n\n@external\ndef foo1(x: int128[2]) -> int128:\n    return self._foo1(x)\n\n@external\ndef bar() -> int128:\n    x: int128[2] = [0, 0]\n    return self._foo0(x)\n\n@external\ndef bar2() -> int128:\n    x: int128[2] = [55, 66]\n    return self._foo0(x)\n\n@external\ndef bar3() -> int128:\n    x: int128[2] = [55, 66]\n    return self._foo1(x)\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.bar() == 0
    assert c.foo1([0, 0]) == 0
    assert c.bar2() == 55
    assert c.bar3() == 66

def test_list_storage_call(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    code = '\ny: int128[2]\n\n@internal\ndef _foo0(x: int128[2]) -> int128:\n    return x[0]\n\n@internal\ndef _foo1(x: int128[2]) -> int128:\n    return x[1]\n\n@external\ndef set():\n    self.y  = [88, 99]\n\n@external\ndef bar0() -> int128:\n    return self._foo0(self.y)\n\n@external\ndef bar1() -> int128:\n    return self._foo1(self.y)\n    '
    c = get_contract_with_gas_estimation(code)
    c.set(transact={})
    assert c.bar0() == 88
    assert c.bar1() == 99

def test_multi_arg_list_call(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\n@internal\ndef _foo0(y: decimal, x: int128[2]) -> int128:\n    return x[0]\n\n@internal\ndef _foo1(x: int128[2], y: decimal) -> int128:\n    return x[1]\n\n@internal\ndef _foo2(y: decimal, x: int128[2]) -> decimal:\n    return y\n\n@internal\ndef _foo3(x: int128[2], y: decimal) -> int128:\n    return x[0]\n\n@internal\ndef _foo4(x: int128[2], y: int128[2]) -> int128:\n    return y[0]\n\n\n@external\ndef foo1(x: int128[2], y: decimal) -> int128:\n    return self._foo1(x, y)\n\n@external\ndef bar() -> int128:\n    x: int128[2] = [0, 0]\n    return self._foo0(0.3434, x)\n\n# list as second parameter\n@external\ndef bar2() -> int128:\n    x: int128[2] = [55, 66]\n    return self._foo0(0.01, x)\n\n@external\ndef bar3() -> decimal:\n    x: int128[2] = [88, 77]\n    return self._foo2(1.33, x)\n\n# list as first parameter\n@external\ndef bar4() -> int128:\n    x: int128[2] = [88, 77]\n    return self._foo1(x, 1.33)\n\n@external\ndef bar5() -> int128:\n    x: int128[2] = [88, 77]\n    return self._foo3(x, 1.33)\n\n# two lists\n@external\ndef bar6() -> int128:\n    x: int128[2] = [88, 77]\n    y: int128[2] = [99, 66]\n    return self._foo4(x, y)\n\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.bar() == 0
    assert c.foo1([0, 0], Decimal('0')) == 0
    assert c.bar2() == 55
    assert c.bar3() == Decimal('1.33')
    assert c.bar4() == 77
    assert c.bar5() == 88

def test_multi_mixed_arg_list_call(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    code = '\n@internal\ndef _fooz(x: int128[2], y: decimal, z: int128[2], a: decimal) -> int128:\n    return z[1]\n\n@internal\ndef _fooa(x: int128[2], y: decimal, z: int128[2], a: decimal) -> decimal:\n    return a\n\n@external\ndef bar() -> (int128, decimal):\n    x: int128[2] = [33, 44]\n    y: decimal = 55.44\n    z: int128[2] = [55, 66]\n    a: decimal = 66.77\n\n    return self._fooz(x, y, z, a), self._fooa(x, y, z, a)\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.bar() == [66, Decimal('66.77')]

def test_internal_function_multiple_lists_as_args(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\n@internal\ndef _foo(y: int128[2], x: Bytes[5]) -> int128:\n    return y[0]\n\n@internal\ndef _foo2(x: Bytes[5], y: int128[2]) -> int128:\n    return y[0]\n\n@external\ndef bar() -> int128:\n    return self._foo([1, 2], b"hello")\n\n@external\ndef bar2() -> int128:\n    return self._foo2(b"hello", [1, 2])\n'
    c = get_contract_with_gas_estimation(code)
    assert c.bar() == 1
    assert c.bar2() == 1

def test_multi_mixed_arg_list_bytes_call(get_contract_with_gas_estimation):
    if False:
        while True:
            i = 10
    code = '\n@internal\ndef _fooz(x: int128[2], y: decimal, z: Bytes[11], a: decimal) -> Bytes[11]:\n    return z\n\n@internal\ndef _fooa(x: int128[2], y: decimal, z: Bytes[11], a: decimal) -> decimal:\n    return a\n\n@internal\ndef _foox(x: int128[2], y: decimal, z: Bytes[11], a: decimal) -> int128:\n    return x[1]\n\n\n@external\ndef bar() -> (Bytes[11], decimal, int128):\n    x: int128[2] = [33, 44]\n    y: decimal = 55.44\n    z: Bytes[11] = b"hello world"\n    a: decimal = 66.77\n\n    return self._fooz(x, y, z, a), self._fooa(x, y, z, a), self._foox(x, y, z, a)\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.bar() == [b'hello world', Decimal('66.77'), 44]
FAILING_CONTRACTS_CALL_VIOLATION = ['\n# should not compile - public to public\n@external\ndef bar() -> int128:\n    return 1\n\n@external\ndef foo() -> int128:\n    return self.bar()\n    ', '\n# should not compile - internal to external\n@external\ndef bar() -> int128:\n    return 1\n\n@internal\ndef _baz() -> int128:\n    return self.bar()\n\n@external\ndef foo() -> int128:\n    return self._baz()\n    ']

@pytest.mark.parametrize('failing_contract_code', FAILING_CONTRACTS_CALL_VIOLATION)
def test_selfcall_call_violation(failing_contract_code, assert_compile_failed):
    if False:
        while True:
            i = 10
    assert_compile_failed(lambda : compile_code(failing_contract_code), CallViolation)
FAILING_CONTRACTS_ARGUMENT_EXCEPTION = ['\n# expected no args, args given\n@internal\ndef bar() -> int128:\n    return 1\n\n@external\ndef foo() -> int128:\n    return self.bar(1)\n    ', '\n# expected args, none given\n@internal\ndef bar(a: int128) -> int128:\n    return 1\n\n@external\ndef foo() -> int128:\n    return self.bar()\n    ', '\n# wrong arg count\n@internal\ndef bar(a: int128) -> int128:\n    return 1\n\n@external\ndef foo() -> int128:\n    return self.bar(1, 2)\n    ', '\n@internal\ndef _foo(x: uint256, y: uint256 = 1):\n    pass\n\n@external\ndef foo(x: uint256, y: uint256):\n    self._foo(x, y=y)\n    ']

@pytest.mark.parametrize('failing_contract_code', FAILING_CONTRACTS_ARGUMENT_EXCEPTION)
def test_selfcall_wrong_arg_count(failing_contract_code, assert_compile_failed):
    if False:
        print('Hello World!')
    assert_compile_failed(lambda : compile_code(failing_contract_code), ArgumentException)
FAILING_CONTRACTS_TYPE_MISMATCH = ['\n# should not compile - value kwarg when calling {0} function\n@{0}\ndef foo():\n    pass\n\n@external\ndef bar():\n    self.foo(value=100)\n    ', '\n# should not compile - gas kwarg when calling {0} function\n@{0}\ndef foo():\n    pass\n\n@external\ndef bar():\n    self.foo(gas=100)\n    ', '\n# should not compile - arbitrary kwargs when calling {0} function\n@{0}\ndef foo():\n    pass\n\n@external\ndef bar():\n    self.foo(baz=100)\n    ', '\n# should not compile - args-as-kwargs to a {0} function\n@{0}\ndef foo(baz: int128):\n    pass\n\n@external\ndef bar():\n    self.foo(baz=100)\n    ']

@pytest.mark.parametrize('failing_contract_code', FAILING_CONTRACTS_TYPE_MISMATCH)
@pytest.mark.parametrize('decorator', ['external', 'internal'])
def test_selfcall_kwarg_raises(failing_contract_code, decorator, assert_compile_failed):
    if False:
        i = 10
        return i + 15
    assert_compile_failed(lambda : compile_code(failing_contract_code.format(decorator)), ArgumentException if decorator == 'internal' else CallViolation)

@pytest.mark.parametrize('i,ln,s,', [(100, 6, 'abcde'), (41, 40, 'a' * 34), (57, 70, 'z' * 68)])
def test_struct_return_1(get_contract_with_gas_estimation, i, ln, s):
    if False:
        print('Hello World!')
    contract = f'\nstruct X:\n    x: int128\n    y: String[{ln}]\n    z: Bytes[{ln}]\n\n@internal\ndef get_struct_x() -> X:\n    return X({{x: {i}, y: "{s}", z: b"{s}"}})\n\n@external\ndef test() -> (int128, String[{ln}], Bytes[{ln}]):\n    ret: X = self.get_struct_x()\n    return ret.x, ret.y, ret.z\n    '
    c = get_contract_with_gas_estimation(contract)
    assert c.test() == [i, s, bytes(s, 'utf-8')]

def test_dynamically_sized_struct_as_arg(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    contract = '\nstruct X:\n    x: uint256\n    y: Bytes[6]\n\n@internal\ndef _foo(x: X) -> Bytes[6]:\n    return x.y\n\n@external\ndef bar() -> Bytes[6]:\n    _X: X = X({x: 1, y: b"hello"})\n    return self._foo(_X)\n    '
    c = get_contract_with_gas_estimation(contract)
    assert c.bar() == b'hello'

def test_dynamically_sized_struct_as_arg_2(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    contract = '\nstruct X:\n    x: uint256\n    y: String[6]\n\n@internal\ndef _foo(x: X) -> String[6]:\n    return x.y\n\n@external\ndef bar() -> String[6]:\n    _X: X = X({x: 1, y: "hello"})\n    return self._foo(_X)\n    '
    c = get_contract_with_gas_estimation(contract)
    assert c.bar() == 'hello'

def test_dynamically_sized_struct_member_as_arg(get_contract_with_gas_estimation):
    if False:
        return 10
    contract = '\nstruct X:\n    x: uint256\n    y: Bytes[6]\n\n@internal\ndef _foo(s: Bytes[6]) -> Bytes[6]:\n    return s\n\n@external\ndef bar() -> Bytes[6]:\n    _X: X = X({x: 1, y: b"hello"})\n    return self._foo(_X.y)\n    '
    c = get_contract_with_gas_estimation(contract)
    assert c.bar() == b'hello'

def test_dynamically_sized_struct_member_as_arg_2(get_contract_with_gas_estimation):
    if False:
        return 10
    contract = '\nstruct X:\n    x: uint256\n    y: String[6]\n\n@internal\ndef _foo(s: String[6]) -> String[6]:\n    return s\n\n@external\ndef bar() -> String[6]:\n    _X: X = X({x: 1, y: "hello"})\n    return self._foo(_X.y)\n    '
    c = get_contract_with_gas_estimation(contract)
    assert c.bar() == 'hello'
st_uint256 = st.integers(min_value=0, max_value=2 ** 256 - 1)
st_string65 = st.text(max_size=65, alphabet=string.printable)
st_bytes65 = st.binary(max_size=65)
st_sarray3 = st.lists(st_uint256, min_size=3, max_size=3)
st_darray3 = st.lists(st_uint256, max_size=3)
internal_call_kwargs_cases = [('uint256', st_uint256), ('String[65]', st_string65), ('Bytes[65]', st_bytes65), ('uint256[3]', st_sarray3), ('DynArray[uint256, 3]', st_darray3)]

@pytest.mark.parametrize('typ1,strategy1', internal_call_kwargs_cases)
@pytest.mark.parametrize('typ2,strategy2', internal_call_kwargs_cases)
def test_internal_call_kwargs(get_contract, typ1, strategy1, typ2, strategy2):
    if False:
        print('Hello World!')

    @given(kwarg1=strategy1, default1=strategy1, kwarg2=strategy2, default2=strategy2)
    @settings(max_examples=5)
    def fuzz(kwarg1, kwarg2, default1, default2):
        if False:
            for i in range(10):
                print('nop')
        code = f'\n@internal\ndef foo(a: {typ1} = {repr(default1)}, b: {typ2} = {repr(default2)}) -> ({typ1}, {typ2}):\n    return a, b\n\n@external\ndef test0() -> ({typ1}, {typ2}):\n    return self.foo()\n\n@external\ndef test1() -> ({typ1}, {typ2}):\n    return self.foo({repr(kwarg1)})\n\n@external\ndef test2() -> ({typ1}, {typ2}):\n    return self.foo({repr(kwarg1)}, {repr(kwarg2)})\n\n@external\ndef test3(x1: {typ1}) -> ({typ1}, {typ2}):\n    return self.foo(x1)\n\n@external\ndef test4(x1: {typ1}, x2: {typ2}) -> ({typ1}, {typ2}):\n    return self.foo(x1, x2)\n        '
        c = get_contract(code)
        assert c.test0() == [default1, default2]
        assert c.test1() == [kwarg1, default2]
        assert c.test2() == [kwarg1, kwarg2]
        assert c.test3(kwarg1) == [kwarg1, default2]
        assert c.test4(kwarg1, kwarg2) == [kwarg1, kwarg2]
    fuzz()