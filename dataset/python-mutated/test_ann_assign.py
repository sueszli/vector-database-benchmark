import pytest
from pytest import raises
from vyper import compiler
from vyper.exceptions import InvalidAttribute, InvalidType, UndeclaredDefinition, UnknownAttribute, VariableDeclarationException
fail_list = [('\n@external\ndef test():\n    a = 1\n    ', UndeclaredDefinition), ('\n@external\ndef test():\n    a = 33.33\n    ', UndeclaredDefinition), ('\n@external\ndef test():\n    a = "test string"\n    ', UndeclaredDefinition), ('\n@external\ndef test():\n    a: int128 = 33.33\n    ', InvalidType), ('\n@external\ndef data() -> int128:\n    s: int128[5] = [1, 2, 3, 4, 5, 6]\n    return 235357\n    ', InvalidType), ('\nstruct S:\n    a: int128\n    b: decimal\n@external\ndef foo() -> int128:\n    s: S = S({a: 1.2, b: 1})\n    return s.a\n    ', InvalidType), ('\nstruct S:\n    a: int128\n    b: decimal\n@external\ndef foo() -> int128:\n    s: S = S({a: 1})\n    ', VariableDeclarationException), ('\nstruct S:\n    a: int128\n    b: decimal\n@external\ndef foo() -> int128:\n    s: S = S({b: 1.2, a: 1})\n    ', InvalidAttribute), ('\nstruct S:\n    a: int128\n    b: decimal\n@external\ndef foo() -> int128:\n    s: S = S({a: 1, b: 1.2, c: 1, d: 33, e: 55})\n    return s.a\n    ', UnknownAttribute), ('\n@external\ndef foo() -> bool:\n    a: uint256 = -1\n    return True\n', InvalidType), ('\n@external\ndef foo() -> bool:\n    a: uint256[2] = [13, -42]\n    return True\n', InvalidType), ('\n@external\ndef foo() -> bool:\n    a: int128 = 170141183460469231731687303715884105728\n    return True\n', InvalidType)]

@pytest.mark.parametrize('bad_code', fail_list)
def test_as_wei_fail(bad_code):
    if False:
        print('Hello World!')
    with raises(bad_code[1]):
        compiler.compile_code(bad_code[0])
valid_list = ['\nstruct S:\n    a: int128\n    b: decimal\n@internal\ndef do_stuff() -> bool:\n    return True\n\n@external\ndef test():\n    a: bool = self.do_stuff() or self.do_stuff()\n    ', '\n@internal\ndef do_stuff() -> bool:\n    return True\n\n@external\ndef test():\n    a: bool = False or self.do_stuff()\n    ', '\n@external\ndef test():\n    a: int128 = 1\n    ', '\n@internal\ndef do_stuff() -> bool:\n    return True\n\n@external\ndef test():\n    a: bool = self.do_stuff()\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_ann_assign_success(good_code):
    if False:
        print('Hello World!')
    assert compiler.compile_code(good_code) is not None