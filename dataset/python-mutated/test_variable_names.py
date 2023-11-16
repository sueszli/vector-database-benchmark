import pytest
from pytest import raises
from vyper import compiler
from vyper.exceptions import NamespaceCollision, StructureException
fail_list = ['\n@external\ndef foo(i: int128) -> int128:\n    varő : int128 = i\n    return varő\n    ', '\n@external\ndef foo(i: int128) -> int128:\n    wei : int128 = i\n    return wei\n    ', '\n@external\ndef foo(i: int128) -> int128:\n    false : int128 = i\n    return false\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_varname_validity_fail(bad_code):
    if False:
        while True:
            i = 10
    with raises(StructureException):
        compiler.compile_code(bad_code)
collision_fail_list = ['\n@external\ndef foo(i: int128) -> int128:\n    int128 : int128 = i\n    return int128\n    ', '\n@external\ndef foo(i: int128) -> int128:\n    decimal : int128 = i\n    return decimal\n    ']

@pytest.mark.parametrize('bad_code', collision_fail_list)
def test_varname_collision_fail(bad_code):
    if False:
        return 10
    with raises(NamespaceCollision):
        compiler.compile_code(bad_code)
valid_list = ['\n@external\ndef foo(i: int128) -> int128:\n    variable : int128 = i\n    return variable\n    ', '\n@external\ndef foo(i: int128) -> int128:\n    var_123 : int128 = i\n    return var_123\n    ', '\n@external\ndef foo(i: int128) -> int128:\n    _var123 : int128 = i\n    return _var123\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_varname_validity_success(good_code):
    if False:
        i = 10
        return i + 15
    assert compiler.compile_code(good_code) is not None