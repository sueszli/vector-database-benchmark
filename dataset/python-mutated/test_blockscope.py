import pytest
from pytest import raises
from vyper import compiler
from vyper.exceptions import NamespaceCollision, UndeclaredDefinition
fail_list = ['\n@external\ndef foo(choice: bool):\n    if (choice):\n        a: int128 = 1\n    a += 1\n    ', '\n@external\ndef foo(choice: bool):\n    if (choice):\n        a: int128 = 0\n    else:\n        a: int128 = 1\n    a += 1\n    ', '\n@external\ndef foo(choice: bool):\n    if (choice):\n        a: int128 = 0\n    else:\n        a += 1\n    ', '\n@external\ndef foo(choice: bool):\n\n    for i in range(4):\n        a: int128 = 0\n    a = 1\n    ', '\n@external\ndef foo(choice: bool):\n\n    for i in range(4):\n        a: int128 = 0\n    a += 1\n    ']

@pytest.mark.parametrize('bad_code', fail_list)
def test_fail_undeclared(bad_code):
    if False:
        return 10
    with raises(UndeclaredDefinition):
        compiler.compile_code(bad_code)
fail_list_collision = ['\n@external\ndef foo():\n    a: int128 = 5\n    a: int128 = 7\n    ']

@pytest.mark.parametrize('bad_code', fail_list_collision)
def test_fail_collision(bad_code):
    if False:
        return 10
    with raises(NamespaceCollision):
        compiler.compile_code(bad_code)
valid_list = ['\n@external\ndef foo(choice: bool, choice2: bool):\n    if (choice):\n        a: int128 = 11\n        if choice2 and a > 1:\n            a -= 1  # should be visible here.\n    ', '\n@external\ndef foo(choice: bool):\n    if choice:\n        a: int128 = 44\n    else:\n        a: uint256 = 42\n    a: bool = True\n    ', '\na: int128\n\n@external\ndef foo():\n    a: int128 = 5\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_valid_blockscope(good_code):
    if False:
        i = 10
        return i + 15
    assert compiler.compile_code(good_code) is not None