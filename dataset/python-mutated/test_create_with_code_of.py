import pytest
from pytest import raises
from vyper import compiler
from vyper.exceptions import ArgumentException, SyntaxException
fail_list = ['\n@external\ndef foo():\n    x: address = create_minimal_proxy_to(\n        0x1234567890123456789012345678901234567890,\n        value=4,\n        value=9\n    )\n    ', '\n@external\ndef foo(_salt: bytes32):\n    x: address = create_minimal_proxy_to(\n        0x1234567890123456789012345678901234567890, salt=keccak256(b"Vyper Rocks!"), salt=_salt\n    )\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_type_mismatch_exception(bad_code):
    if False:
        i = 10
        return i + 15
    with raises((SyntaxException, ArgumentException)):
        compiler.compile_code(bad_code)
valid_list = ['\n@external\ndef foo():\n    # test that create_forwarder_to is valid syntax; the name is deprecated\n    x: address = create_forwarder_to(0x1234567890123456789012345678901234567890)\n    ', '\n@external\ndef foo():\n    x: address = create_minimal_proxy_to(0x1234567890123456789012345678901234567890)\n    ', '\n@external\ndef foo():\n    x: address = create_minimal_proxy_to(\n        0x1234567890123456789012345678901234567890,\n        value=as_wei_value(9, "wei")\n    )\n    ', '\n@external\ndef foo():\n    x: address = create_minimal_proxy_to(0x1234567890123456789012345678901234567890, value=9)\n    ', '\n@external\ndef foo():\n    x: address = create_minimal_proxy_to(\n        0x1234567890123456789012345678901234567890,\n        salt=keccak256(b"Vyper Rocks!")\n    )\n    ', '\n@external\ndef foo(_salt: bytes32):\n    x: address = create_minimal_proxy_to(0x1234567890123456789012345678901234567890, salt=_salt)\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_rlp_success(good_code):
    if False:
        return 10
    assert compiler.compile_code(good_code) is not None