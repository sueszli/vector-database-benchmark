import pytest
from vyper import compiler
from vyper.exceptions import InvalidLiteral, InvalidType, TypeMismatch
fail_list = [('\nbar: int128[3][3]\n@external\ndef foo():\n    self.bar = [[1, 2], [3, 4, 5], [6, 7, 8]]\n    ', InvalidType), ('\nbar: int128[3][3]\n@external\ndef foo():\n    self.bar = [[1, 2, 3], [4, 5, 6], [7.0, 8.0, 9.0]]\n    ', InvalidLiteral), ('\n@external\ndef foo() -> int128[2]:\n    return [[1,2],[3,4]]\n    ', InvalidType), ('\n@external\ndef foo() -> int128[2][2]:\n    return [1,2]\n    ', InvalidType), ('\ny: address[2][2]\n\n@external\ndef foo(x: int128[2][2]) -> int128:\n    self.y = x\n    return 768\n    ', TypeMismatch)]

@pytest.mark.parametrize('bad_code,exc', fail_list)
def test_nested_list_fail(bad_code, exc):
    if False:
        while True:
            i = 10
    with pytest.raises(exc):
        compiler.compile_code(bad_code)
valid_list = ['\nbar: int128[3][3]\n@external\ndef foo():\n    self.bar = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n    ', '\nbar: decimal[3][3]\n@external\ndef foo():\n    self.bar = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_nested_list_sucess(good_code):
    if False:
        return 10
    assert compiler.compile_code(good_code) is not None