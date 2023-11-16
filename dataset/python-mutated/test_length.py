import pytest

def test_test_length(get_contract_with_gas_estimation):
    if False:
        for i in range(10):
            print('nop')
    test_length = '\ny: Bytes[10]\n\n@external\ndef foo(inp: Bytes[10]) -> uint256:\n    x: Bytes[5] = slice(inp,1, 5)\n    self.y = slice(inp, 2, 4)\n    return len(inp) * 100 + len(x) * 10 + len(self.y)\n    '
    c = get_contract_with_gas_estimation(test_length)
    assert c.foo(b'badminton') == 954, c.foo(b'badminton')
    print('Passed length test')

@pytest.mark.parametrize('typ', ['DynArray[uint256, 50]', 'Bytes[50]', 'String[50]'])
def test_zero_length(get_contract_with_gas_estimation, typ):
    if False:
        while True:
            i = 10
    code = f'\n@external\ndef boo() -> uint256:\n    e: uint256 = len(empty({typ}))\n    return e\n    '
    c = get_contract_with_gas_estimation(code)
    assert c.boo() == 0