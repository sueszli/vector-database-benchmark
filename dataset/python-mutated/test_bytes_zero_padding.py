import hypothesis
import pytest

@pytest.fixture(scope='module')
def little_endian_contract(get_contract_module):
    if False:
        return 10
    code = '\n@internal\n@view\ndef to_little_endian_64(_value: uint256) -> Bytes[8]:\n    y: uint256 = 0\n    x: uint256 = _value\n    for _ in range(8):\n        y = (y << 8) | (x & 255)\n        x >>= 8\n    return slice(convert(y, bytes32), 24, 8)\n\n@external\n@view\ndef get_count(counter: uint256) -> Bytes[24]:\n    return self.to_little_endian_64(counter)\n    '
    c = get_contract_module(code)
    return c

@pytest.mark.fuzzing
@hypothesis.given(value=hypothesis.strategies.integers(min_value=0, max_value=2 ** 64))
def test_zero_pad_range(little_endian_contract, value):
    if False:
        print('Hello World!')
    actual_bytes = value.to_bytes(8, byteorder='little')
    contract_bytes = little_endian_contract.get_count(value)
    assert contract_bytes == actual_bytes