import pytest
from vyper import compiler
from vyper.exceptions import ArgumentException
fail_list = ['\n@external\ndef foo():\n    x = as_wei_value(5, "vader")\n    ', '\n@external\ndef foo(x: int128, x: int128): pass\n    ', '\n@external\ndef foo(x): pass\n    ', '\n@external\ndef foo() -> int128:\n    return as_wei_value(10)\n    ', '\n@external\ndef foo():\n    x: bytes32 = keccak256("moose", 3)\n    ', '\n@external\ndef foo():\n    x: Bytes[4] = raw_call(0x1234567890123456789012345678901234567890, outsize=4)\n    ', '\n@external\ndef foo():\n    x: Bytes[4] = raw_call(\n        0x1234567890123456789012345678901234567890, b"cow", gas=111111, outsize=4, moose=9\n    )\n    ', '\n@external\ndef foo():\n    x: Bytes[4] = create_minimal_proxy_to(0x1234567890123456789012345678901234567890, outsize=4)\n    ', '\nx: public()\n    ', '\n@external\ndef foo():\n    raw_log([], b"cow", "dog")\n    ', '\n@external\ndef foo():\n    x: Bytes[10] = concat(b"")\n    ', '\n@external\ndef foo():\n    x: Bytes[4] = create_minimal_proxy_to(0x1234567890123456789012345678901234567890, b"cow")\n    ', '\n@external\ndef foo():\n    a: uint256 = min()\n    ', '\n@external\ndef foo():\n    a: uint256 = min(1)\n    ', '\n@external\ndef foo():\n    a: uint256 = min(1, 2, 3)\n    ', '\n@external\ndef foo():\n    for i in range():\n        pass\n    ', '\n@external\ndef foo():\n    for i in range(1, 2, 3, 4):\n        pass\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_function_declaration_exception(bad_code):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ArgumentException):
        compiler.compile_code(bad_code)