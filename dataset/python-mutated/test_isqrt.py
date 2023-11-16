import math
import hypothesis
import pytest
from vyper.utils import SizeLimits

@pytest.fixture(scope='module')
def isqrt_contract(get_contract_module):
    if False:
        for i in range(10):
            print('nop')
    code = '\n@external\ndef test(a: uint256) -> uint256:\n    return isqrt(a)\n    '
    c = get_contract_module(code)
    return c

def test_isqrt_literal(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    val = 2
    code = f'\n@external\ndef test() -> uint256:\n    return isqrt({val})\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.test() == math.isqrt(val)

def test_isqrt_variable(get_contract_with_gas_estimation):
    if False:
        return 10
    code = '\n@external\ndef test(a: uint256) -> uint256:\n    return isqrt(a)\n    '
    c = get_contract_with_gas_estimation(code)
    val = 3333
    assert c.test(val) == math.isqrt(val)
    val = 10 ** 17
    assert c.test(val) == math.isqrt(val)
    assert c.test(0) == 0

def test_isqrt_internal_variable(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    val = 44001
    code = f'\n@external\ndef test2() -> uint256:\n    a: uint256 = {val}\n    return isqrt(a)\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.test2() == math.isqrt(val)

def test_isqrt_storage(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = '\ns_var: uint256\n\n@external\ndef test(a: uint256) -> uint256:\n    self.s_var = a + 1\n    return isqrt(self.s_var)\n    '
    c = get_contract_with_gas_estimation(code)
    val = 1221
    assert c.test(val) == math.isqrt(val + 1)
    val = 10001
    assert c.test(val) == math.isqrt(val + 1)

def test_isqrt_storage_internal_variable(get_contract_with_gas_estimation):
    if False:
        i = 10
        return i + 15
    val = 44444
    code = f'\ns_var: uint256\n\n@external\ndef test2() -> uint256:\n    self.s_var = {val}\n    return isqrt(self.s_var)\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.test2() == math.isqrt(val)

def test_isqrt_inline_memory_correct(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    code = "\n@external\ndef test(a: uint256) -> (uint256, uint256, uint256, uint256, uint256, String[100]):\n    x: uint256 = 1\n    y: uint256 = 2\n    z: uint256 = 3\n    e: uint256 = isqrt(a)\n    f: String[100] = 'hello world'\n    return a, x, y, z, e, f\n    "
    c = get_contract_with_gas_estimation(code)
    val = 21
    assert c.test(val) == [val, 1, 2, 3, math.isqrt(val), 'hello world']

@pytest.mark.fuzzing
@hypothesis.given(value=hypothesis.strategies.integers(min_value=0, max_value=SizeLimits.MAX_UINT256))
@hypothesis.example(SizeLimits.MAX_UINT256)
@hypothesis.example(0)
@hypothesis.example(1)
@hypothesis.example(2704)
@hypothesis.example(110889)
@hypothesis.example(32239684)
def test_isqrt_valid_range(isqrt_contract, value):
    if False:
        for i in range(10):
            print('nop')
    vyper_isqrt = isqrt_contract.test(value)
    actual_isqrt = math.isqrt(value)
    assert vyper_isqrt == actual_isqrt
    next = vyper_isqrt + 1
    assert vyper_isqrt * vyper_isqrt <= value
    assert next * next > value