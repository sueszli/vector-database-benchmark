import math
import pytest
VALID_BITS = list(range(8, 257, 8))

@pytest.mark.parametrize('bits', VALID_BITS)
def test_mkstr(get_contract_with_gas_estimation, bits):
    if False:
        i = 10
        return i + 15
    n_digits = math.ceil(bits * math.log(2) / math.log(10))
    code = f'\n@external\ndef foo(inp: uint{bits}) -> String[{n_digits}]:\n    return uint2str(inp)\n    '
    c = get_contract_with_gas_estimation(code)
    for i in [1, 2, 2 ** bits - 1, 0]:
        assert c.foo(i) == str(i), (i, c.foo(i))

@pytest.mark.parametrize('bits', VALID_BITS)
def test_mkstr_buffer(get_contract, bits):
    if False:
        return 10
    n_digits = math.ceil(bits * math.log(2) / math.log(10))
    code = f'\nsome_string: String[{n_digits}]\n@internal\ndef _foo(x: uint{bits}):\n    self.some_string = uint2str(x)\n\n@external\ndef foo(x: uint{bits}) -> uint256:\n    y: uint256 = 0\n    self._foo(x)\n    return y\n    '
    c = get_contract(code)
    assert c.foo(2 ** bits - 1) == 0, bits