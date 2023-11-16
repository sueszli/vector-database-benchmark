import pytest
from pytest import raises
from vyper import compiler
from vyper.exceptions import NonPayableViolation
fail_list = ['\n@external\ndef foo():\n    x: uint256 = msg.value\n']

@pytest.mark.parametrize('bad_code', fail_list)
def test_variable_decleration_exception(bad_code):
    if False:
        i = 10
        return i + 15
    with raises(NonPayableViolation):
        compiler.compile_code(bad_code)
valid_list = ['\nx: int128\n@external\n@payable\ndef foo() -> int128:\n    self.x = 5\n    return self.x\n    ', '\n@external\n@payable\ndef foo():\n    x: uint256 = msg.value\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_block_success(good_code):
    if False:
        for i in range(10):
            print('nop')
    assert compiler.compile_code(good_code) is not None