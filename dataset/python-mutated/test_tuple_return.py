import pytest

@pytest.mark.parametrize('string', ['a', 'abc', 'abcde', 'potato'])
def test_string_inside_tuple(get_contract, string):
    if False:
        for i in range(10):
            print('nop')
    code = f'\n@external\ndef test_return() -> (String[6], uint256):\n    return "{string}", 42\n    '
    c1 = get_contract(code)
    code = '\ninterface jsonabi:\n    def test_return() -> (String[6], uint256): view\n\n@external\ndef test_values(a: address) -> (String[6], uint256):\n    return jsonabi(a).test_return()\n    '
    c2 = get_contract(code)
    assert c2.test_values(c1.address) == [string, 42]

@pytest.mark.parametrize('string', ['a', 'abc', 'abcde', 'potato'])
def test_bytes_inside_tuple(get_contract, string):
    if False:
        for i in range(10):
            print('nop')
    code = f'\n@external\ndef test_return() -> (Bytes[6], uint256):\n    return b"{string}", 42\n    '
    c1 = get_contract(code)
    code = '\ninterface jsonabi:\n    def test_return() -> (Bytes[6], uint256): view\n\n@external\ndef test_values(a: address) -> (Bytes[6], uint256):\n    return jsonabi(a).test_return()\n    '
    c2 = get_contract(code)
    assert c2.test_values(c1.address) == [bytes(string, 'utf-8'), 42]