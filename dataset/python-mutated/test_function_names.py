import pytest
from pytest import raises
from vyper import compiler
from vyper.exceptions import NamespaceCollision, StructureException
fail_list = ['\n@external\ndef ő1qwerty(i: int128) -> int128:\n    temp_var : int128 = i\n    return temp_var\n    ', '\n@external\ndef false(i: int128) -> int128:\n    temp_var : int128 = i\n    return temp_var\n    ', '\n@external\ndef wei(i: int128) -> int128:\n    temp_var : int128 = i\n    return temp_var1\n    ', '\nfoo: public(uint256)\n\n@external\ndef foo():\n    pass\n    ', '\n@external\ndef foo():\n    pass\n\nfoo: public(uint256)\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_varname_validity_fail(bad_code):
    if False:
        return 10
    with raises((StructureException, NamespaceCollision)):
        compiler.compile_code(bad_code)
valid_list = ['\n@external\ndef func(i: int128) -> int128:\n    variable : int128 = i\n    return variable\n    ', '\n@external\ndef func_to_do_math(i: int128) -> int128:\n    var_123 : int128 = i\n    return var_123\n    ', '\n@external\ndef first1(i: int128) -> int128:\n    _var123 : int128 = i\n    return _var123\n    ', '\n@external\ndef int128(i: int128) -> int128:\n    temp_var : int128 = i\n    return temp_var\n    ', '\n@external\ndef decimal(i: int128) -> int128:\n    temp_var : int128 = i\n    return temp_var\n    ', '\n@external\ndef floor():\n    pass\n    ', '\n@internal\ndef append():\n    pass\n\n@external\ndef foo():\n    self.append()\n    ', '\n@internal\n@view\ndef gfah():\n    pass\n\n@internal\n@view\ndef eexo():\n    pass\n    ', '\n@internal\n@view\ndef gfah():\n    pass\n\n@external\n@view\ndef eexo():\n    pass\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_varname_validity_success(good_code):
    if False:
        print('Hello World!')
    assert compiler.compile_code(good_code) is not None