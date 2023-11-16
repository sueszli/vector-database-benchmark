import pytest
from vyper import compiler
from vyper.exceptions import InvalidType
fail_list = ['\n@external\ndef foo():\n    selfdestruct(7)\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_block_fail(bad_code):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(InvalidType):
        compiler.compile_code(bad_code)
valid_list = ['\n@external\ndef foo():\n    selfdestruct(0x1234567890123456789012345678901234567890)\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_block_success(good_code):
    if False:
        for i in range(10):
            print('nop')
    assert compiler.compile_code(good_code) is not None