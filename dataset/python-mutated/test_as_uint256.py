import pytest
from vyper import compiler
from vyper.exceptions import InvalidType, TypeMismatch
fail_list = [('\n@external\ndef convert2(inp: uint256) -> uint256:\n    return convert(inp, bytes32)\n    ', TypeMismatch), ('\n@external\ndef modtest(x: uint256, y: int128) -> uint256:\n    return x % y\n    ', TypeMismatch), ('\n@internal\ndef ret_non():\n    pass\n\n@external\ndef test():\n    a: uint256 = 100 * self.ret_non()\n    ', InvalidType)]

@pytest.mark.parametrize('bad_code,exc', fail_list)
def test_as_uint256_fail(bad_code, exc):
    if False:
        return 10
    with pytest.raises(exc):
        compiler.compile_code(bad_code)