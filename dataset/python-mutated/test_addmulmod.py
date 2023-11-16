import pytest
from vyper.exceptions import InvalidType
fail_list = [('\n@external\ndef foo() -> uint256:\n    return uint256_addmod(1.1, 1.2, 3.0)\n    ', InvalidType), ('\n@external\ndef foo() -> uint256:\n    return uint256_mulmod(1.1, 1.2, 3.0)\n    ', InvalidType)]

@pytest.mark.parametrize('code,exc', fail_list)
def test_add_mod_fail(assert_compile_failed, get_contract, code, exc):
    if False:
        while True:
            i = 10
    assert_compile_failed(lambda : get_contract(code), exc)