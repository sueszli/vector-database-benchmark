import pytest
from vyper.exceptions import SyntaxException
fail_list = ['\nx: Bytes[1:3]\n    ', '\nb: int128[int128: address]\n    ', '\nx: int128[5]\n@external\ndef foo():\n    self.x[2:4] = 3\n    ', '\nx: int128[5]\n@external\ndef foo():\n    z = self.x[2:4]\n    ', '\n@external\ndef foo():\n    x: int128[5]\n    z = x[2:4]\n    ', '\nTransfer: event({_rom&: indexed(address)})\n    ', '\n@external\ndef test() -> uint256:\n    for i in range(0, 4):\n      return 0\n    else:\n      return 1\n    return 1\n    ', '\n@external\ndef foo():\n    x = y = 3\n    ', '\n@external\ndef foo():\n    x: address = create_minimal_proxy_to(0x123456789012345678901234567890123456789)\n    ', '\n@external\ndef foo():\n    x: Bytes[4] = raw_call(0x123456789012345678901234567890123456789, "cow", max_outsize=4)\n    ', '\n@external\ndef foo():\n    x: address = 0x12345678901234567890123456789012345678901\n    ', '\n@external\ndef foo():\n    x: address = 0x01234567890123456789012345678901234567890\n    ', '\n@external\ndef foo():\n    x: address = 0x123456789012345678901234567890123456789\n    ', '\na: internal(uint256)\n    ', '\n@external\ndef foo():\n    x: uint256 = +1  # test UAdd ast blocked\n    ', '\n@internal\ndef f(a:uint256,/):  # test posonlyargs blocked\n    return\n\n@external\ndef g():\n    self.f()\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_syntax_exception(assert_compile_failed, get_contract, bad_code):
    if False:
        i = 10
        return i + 15
    assert_compile_failed(lambda : get_contract(bad_code), SyntaxException)