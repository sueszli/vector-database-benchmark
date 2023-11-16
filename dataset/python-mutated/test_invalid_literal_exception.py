import pytest
from pytest import raises
from vyper import compiler
from vyper.exceptions import InvalidLiteral
fail_list = ['\nb: decimal\n@external\ndef foo():\n    self.b = 7.5178246872145875217495129745982164981654986129846\n    ', '\n@external\ndef foo():\n    x: uint256 = convert(-(-(-1)), uint256)\n    ', '\n@external\ndef foo(x: int128):\n    y: int128 = 7\n    for i in range(x, x + y):\n        pass\n    ', '\n@external\ndef foo():\n    x: String[100] = "these bytes are nо gооd because the o\'s are from the Russian alphabet"\n    ', '\n@external\ndef foo():\n    x: String[100] = "这个傻老外不懂中文"\n    ', '\n@external\ndef foo():\n    a: Bytes[100] = "ѓtest"\n    ', '\n@external\ndef foo():\n    a: bytes32 = keccak256("ѓtest")\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_invalid_literal_exception(bad_code):
    if False:
        print('Hello World!')
    with raises(InvalidLiteral):
        compiler.compile_code(bad_code)