import itertools
import pytest

def test_bytes_literal_code(get_contract_with_gas_estimation):
    if False:
        print('Hello World!')
    bytes_literal_code = '\n@external\ndef foo() -> Bytes[5]:\n    return b"horse"\n\n@external\ndef bar() -> Bytes[10]:\n    return concat(b"b", b"a", b"d", b"m", b"i", b"", b"nton")\n\n@external\ndef baz() -> Bytes[40]:\n    return concat(b"0123456789012345678901234567890", b"12")\n\n@external\ndef baz2() -> Bytes[40]:\n    return concat(b"01234567890123456789012345678901", b"12")\n\n@external\ndef baz3() -> Bytes[40]:\n    return concat(b"0123456789012345678901234567890", b"1")\n\n@external\ndef baz4() -> Bytes[100]:\n    return concat(b"01234567890123456789012345678901234567890123456789",\n                  b"01234567890123456789012345678901234567890123456789")\n    '
    c = get_contract_with_gas_estimation(bytes_literal_code)
    assert c.foo() == b'horse'
    assert c.bar() == b'badminton'
    assert c.baz() == b'012345678901234567890123456789012'
    assert c.baz2() == b'0123456789012345678901234567890112'
    assert c.baz3() == b'01234567890123456789012345678901'
    assert c.baz4() == b'0123456789' * 10
    print('Passed string literal test')

@pytest.mark.parametrize('i,e,_s', itertools.product([95, 96, 97], [63, 64, 65], [31, 32, 33]))
def test_bytes_literal_splicing_fuzz(get_contract_with_gas_estimation, i, e, _s):
    if False:
        print('Hello World!')
    kode = f'''\nmoo: Bytes[100]\n\n@external\ndef foo(s: uint256, L: uint256) -> Bytes[100]:\n    x: int128 = 27\n    r: Bytes[100] = slice(b"{'c' * i}", s, L)\n    y: int128 = 37\n    if x * y == 999:\n        return r\n    return b"3434346667777"\n\n@external\ndef bar(s: uint256, L: uint256) -> Bytes[100]:\n    self.moo = b"{'c' * i}"\n    x: int128 = 27\n    r: Bytes[100] = slice(self.moo, s, L)\n    y: int128  = 37\n    if x * y == 999:\n        return r\n    return b"3434346667777"\n\n@external\ndef baz(s: uint256, L: uint256) -> Bytes[100]:\n    x: int128 = 27\n    self.moo = slice(b"{'c' * i}", s, L)\n    y: int128 = 37\n    if x * y == 999:\n        return self.moo\n    return b"3434346667777"\n    '''
    c = get_contract_with_gas_estimation(kode)
    o1 = c.foo(_s, e - _s)
    o2 = c.bar(_s, e - _s)
    o3 = c.baz(_s, e - _s)
    assert o1 == o2 == o3 == b'c' * (e - _s), (i, _s, e - _s, o1, o2, o3)
    print('Passed string literal splicing fuzz-test')