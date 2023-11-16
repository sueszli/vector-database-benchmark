import pytest
from vyper import compiler
from vyper.exceptions import StructureException
fail_list = ['\n@external\ndef foo() -> int128:\n    x: int128 = 45\n    return x.codesize\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_block_fail(bad_code):
    if False:
        i = 10
        return i + 15
    with pytest.raises(StructureException):
        compiler.compile_code(bad_code)
valid_list = ['\n@external\ndef foo() -> uint256:\n    x: address = 0x1234567890123456789012345678901234567890\n    return x.codesize\n    ', '\n@external\ndef foo() -> uint256:\n    return self.codesize\n    ', '\nstruct Foo:\n    t: address\nfoo: Foo\n\n@external\ndef bar() -> uint256:\n    return self.foo.t.codesize\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_block_success(good_code):
    if False:
        print('Hello World!')
    assert compiler.compile_code(good_code) is not None