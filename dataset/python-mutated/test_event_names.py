import pytest
from pytest import raises
from vyper import compiler
from vyper.exceptions import InvalidType, NamespaceCollision, StructureException, SyntaxException
fail_list = [('\nevent Âssign:\n    variable: int128\n\n@external\ndef foo(i: int128) -> int128:\n    temp_var : int128 = i\n    log Âssign(temp_var)\n    return temp_var\n    ', StructureException), ('\nevent int128:\n    variable: int128\n\n@external\ndef foo(i: int128) -> int128:\n    temp_var : int128 = i\n    log int128(temp_var)\n    return temp_var\n    ', NamespaceCollision), ('\nevent decimal:\n    variable: int128\n\n@external\ndef foo(i: int128) -> int128:\n    temp_var : int128 = i\n    log decimal(temp_var)\n    return temp_var\n    ', NamespaceCollision), ('\nevent wei:\n    variable: int128\n\n@external\ndef foo(i: int128) -> int128:\n    temp_var : int128 = i\n    log wei(temp_var)\n    return temp_var\n    ', StructureException), ('\nevent false:\n    variable: int128\n\n@external\ndef foo(i: int128) -> int128:\n    temp_var : int128 = i\n    log false(temp_var)\n    return temp_var\n    ', StructureException), ('\nTransfer: eve.t({_from: indexed(address)})\n    ', SyntaxException), ('\nevent Transfer:\n    _from: i.dexed(address)\n    _to: indexed(address)\n    lue: uint256\n    ', InvalidType)]

@pytest.mark.parametrize('bad_code,exc', fail_list)
def test_varname_validity_fail(bad_code, exc):
    if False:
        return 10
    with raises(exc):
        compiler.compile_code(bad_code)
valid_list = ['\nevent Assigned:\n    variable: int128\n\n@external\ndef foo(i: int128) -> int128:\n    variable : int128 = i\n    log Assigned(variable)\n    return variable\n    ', '\nevent _Assign:\n    variable: int128\n\n@external\ndef foo(i: int128) -> int128:\n    variable : int128 = i\n    log _Assign(variable)\n    return variable\n    ', '\nevent Assigned1:\n    variable: int128\n\n@external\ndef foo(i: int128) -> int128:\n    variable : int128 = i\n    log Assigned1(variable)\n    return variable\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_varname_validity_success(good_code):
    if False:
        return 10
    assert compiler.compile_code(good_code) is not None