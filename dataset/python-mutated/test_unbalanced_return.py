import pytest
from vyper import compiler
from vyper.exceptions import FunctionDeclarationException, StructureException
fail_list = [('\n@external\ndef foo() -> int128:\n    pass\n    ', FunctionDeclarationException), ('\n@external\ndef foo() -> int128:\n    if False:\n        return 123\n    ', FunctionDeclarationException), ('\n@external\ndef test() -> int128:\n    if 1 == 1 :\n        return 1\n        if True:\n            return 0\n    else:\n        assert msg.sender != msg.sender\n    ', FunctionDeclarationException), ('\n@internal\ndef valid_address(sender: address) -> bool:\n    selfdestruct(sender)\n    return True\n    ', StructureException), ('\n@internal\ndef valid_address(sender: address) -> bool:\n    selfdestruct(sender)\n    a: address = sender\n    ', StructureException), ('\n@internal\ndef valid_address(sender: address) -> bool:\n    if sender == empty(address):\n        selfdestruct(sender)\n        _sender: address = sender\n    else:\n        return False\n    ', StructureException), ('\n@internal\ndef foo() -> bool:\n    raw_revert(b"vyper")\n    return True\n    ', StructureException), ('\n@internal\ndef foo() -> bool:\n    raw_revert(b"vyper")\n    x: uint256 = 3\n    ', StructureException), ('\n@internal\ndef foo(x: uint256) -> bool:\n    if x == 2:\n        raw_revert(b"vyper")\n        a: uint256 = 3\n    else:\n        return False\n    ', StructureException)]

@pytest.mark.parametrize('bad_code,exc', fail_list)
def test_return_mismatch(bad_code, exc):
    if False:
        print('Hello World!')
    with pytest.raises(exc):
        compiler.compile_code(bad_code)
valid_list = ['\n@external\ndef foo() -> int128:\n    return 123\n    ', '\n@external\ndef foo() -> int128:\n    if True:\n        return 123\n    else:\n        raise "test"\n    ', '\n@external\ndef foo() -> int128:\n    if False:\n        return 123\n    else:\n        selfdestruct(msg.sender)\n    ', '\n@external\ndef foo() -> int128:\n    if False:\n        return 123\n    return 333\n    ', '\n@external\ndef test() -> int128:\n    if 1 == 1 :\n        return 1\n    else:\n        assert msg.sender != msg.sender\n        return 0\n    ', '\n@external\ndef test() -> int128:\n    x: bytes32 = empty(bytes32)\n    if False:\n        if False:\n            return 0\n        else:\n            x = keccak256(x)\n            return 1\n    else:\n        x = keccak256(x)\n        return 1\n    return 1\n    ', '\n@external\ndef foo() -> int128:\n    if True:\n        return 123\n    else:\n        raw_revert(b"vyper")\n    ']

@pytest.mark.parametrize('good_code', valid_list)
def test_return_success(good_code):
    if False:
        for i in range(10):
            print('nop')
    assert compiler.compile_code(good_code) is not None